import copy
import os
import resource
import signal
from collections.abc import Mapping
from typing import Any, Dict, List, Optional, Tuple

import hydra
import lightning as L
import torch
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from lightning.pytorch.plugins.environments import SLURMEnvironment
from omegaconf import DictConfig, OmegaConf, open_dict
from tabulate import tabulate

from egomimic.rldb.zarr.utils import DataSchematic, set_global_seed
from egomimic.scripts.evaluation.eval import Eval
from egomimic.utils.aws.aws_data_utils import load_env
from egomimic.utils.instantiators import instantiate_callbacks, instantiate_loggers
from egomimic.utils.logging_utils import log_hyperparameters
from egomimic.utils.pylogger import RankedLogger
from egomimic.utils.utils import extras, task_wrapper

OmegaConf.register_new_resolver("eval", eval)
log = RankedLogger(__name__, rank_zero_only=True)


def _log_dataset_frame_counts(train_datasets: dict, valid_datasets: dict) -> None:
    rows = []
    for name, ds in train_datasets.items():
        rows.append(("train", name, len(ds)))
    if train_datasets:
        rows.append(
            ("TOTAL", "(train)", sum(len(ds) for ds in train_datasets.values()))
        )
    for name, ds in valid_datasets.items():
        rows.append(("valid", name, len(ds)))
    if valid_datasets:
        rows.append(
            ("TOTAL", "(valid)", sum(len(ds) for ds in valid_datasets.values()))
        )
    table = tabulate(
        rows,
        headers=["Split", "Dataset", "Frames"],
        tablefmt="rounded_outline",
        intfmt=",",
    )
    log.info("Dataset frame counts:\n" + table)


def _dataset_cfg_with_skip_videos(dataset_cfg: DictConfig) -> DictConfig:
    dataset_cfg = copy.deepcopy(dataset_cfg)
    with open_dict(dataset_cfg):
        dataset_cfg.skip_videos = True
    return dataset_cfg


def _env_value_is_enabled(name: str) -> bool:
    value = os.environ.get(name)
    if value is None:
        return False
    return value.strip().lower() not in {"", "0", "false", "no", "none", "off"}


def _raise_open_file_limit() -> None:
    target_value = os.environ.get("EGOVERSE_NOFILE_LIMIT", "65536")
    try:
        target = int(target_value)
    except ValueError:
        log.warning(f"Ignoring invalid EGOVERSE_NOFILE_LIMIT={target_value!r}.")
        return

    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        if hard != resource.RLIM_INFINITY:
            target = min(target, hard)
        if soft < target:
            resource.setrlimit(resource.RLIMIT_NOFILE, (target, hard))
            soft = target
        log.info(f"Open-file limit: soft={soft}, hard={hard}.")
    except (OSError, ValueError) as exc:
        log.warning(f"Could not raise open-file limit: {exc}")


@task_wrapper
def train(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Trains the model. Can additionally evaluate on a testset, using best weights obtained during
    training.

    This method is wrapped in optional @task_wrapper decorator, that controls the behavior during
    failure. Useful for multiruns, saving info about the crash, etc.

    :param cfg: A DictConfig configuration composed by Hydra.
    :return: A tuple with metrics and dict with all instantiated objects.
    """
    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

        set_global_seed(cfg.seed)
    else:
        raise ValueError("Seed must be provided in cfg for reproducibility!")

    _raise_open_file_limit()

    sharing_strategy = os.environ.get("EGOVERSE_MP_SHARING_STRATEGY", "file_system")
    if sharing_strategy:
        try:
            torch.multiprocessing.set_sharing_strategy(sharing_strategy)
            log.info(
                f"Set torch multiprocessing sharing strategy to {sharing_strategy!r}."
            )
        except RuntimeError as exc:
            log.warning(
                f"Could not set torch multiprocessing sharing strategy "
                f"to {sharing_strategy!r}: {exc}"
            )

    matmul_precision = os.environ.get("EGOVERSE_MATMUL_PRECISION", "high").lower()
    if matmul_precision in {"highest", "high", "medium"}:
        torch.set_float32_matmul_precision(matmul_precision)
        log.info(f"Set torch float32 matmul precision to {matmul_precision!r}.")
    elif matmul_precision not in {"", "none", "false", "0"}:
        log.warning(
            f"Ignoring invalid EGOVERSE_MATMUL_PRECISION={matmul_precision!r}; "
            "expected one of highest/high/medium."
        )

    if os.environ.get("EGOVERSE_CUDNN_BENCHMARK", "1").lower() in {
        "1",
        "true",
        "yes",
    }:
        torch.backends.cudnn.benchmark = True
        log.info("Enabled torch.backends.cudnn.benchmark.")

    load_env()
    # log.info(f"Instantiating data schematic <{cfg.data_schematic._target_}>")

    data_schematic: DataSchematic = hydra.utils.instantiate(cfg.data_schematic)

    # Modify dataset configs to include `data_schematic` dynamically at runtime
    train_datasets = {}
    for dataset_name in cfg.data.train_datasets:
        train_datasets[dataset_name] = hydra.utils.instantiate(
            cfg.data.train_datasets[dataset_name]
        )

    valid_datasets = {}
    for dataset_name in cfg.data.valid_datasets:
        valid_datasets[dataset_name] = hydra.utils.instantiate(
            cfg.data.valid_datasets[dataset_name]
        )

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    assert (
        "MultiDataModuleWrapper" in cfg.data._target_
    ), "cfg.data._target_ must be 'MultiDataModuleWrapper'"
    datamodule: LightningDataModule = hydra.utils.instantiate(
        cfg.data, train_datasets=train_datasets, valid_datasets=valid_datasets
    )

    # TODO: deprecate shape inference in favor of LeRobotDatasetMetadata
    # NOTE: We assume that each dataset is of a unique embodiment. Multi-task datasets should be wrapped around TODO: MultiRLDBDataset

    for dataset_name, dataset in datamodule.train_datasets.items():
        log.info(f"Inferring shapes for dataset <{dataset_name}>")
        # Avoid touching video decoders in the main process before DataLoader
        # workers are created. LeRobot warns that this can poison worker-side
        # video loading; for shape inference we only need proprio/action keys.
        shape_dataset = hydra.utils.instantiate(
            _dataset_cfg_with_skip_videos(cfg.data.train_datasets[dataset_name])
        )
        data_schematic.infer_shapes_from_batch(shape_dataset[0])
        # instantiate norm datasets which is same as dataset but with keymap without the image keys
        instantiate_copy = copy.deepcopy(cfg.data.train_datasets[dataset_name])
        # Support both config layouts:
        #   - old style: resolver.key_map (viperx)
        #   - new style: key_map directly (aria)
        if OmegaConf.select(instantiate_copy, "resolver.key_map") is not None:
            keymap_cfg = instantiate_copy.resolver.key_map
        elif OmegaConf.select(instantiate_copy, "key_map") is not None:
            keymap_cfg = instantiate_copy.key_map
        else:
            keymap_cfg = OmegaConf.create({})

        km = OmegaConf.to_container(keymap_cfg, resolve=False)

        # Strip camera keys from norm dataset (norm only needs proprio + action)
        def _looks_like_image_key(key: Any, value: Any) -> bool:
            candidates = [str(key)]
            if isinstance(value, Mapping):
                candidates.extend(str(v) for v in value.values())
                if value.get("key_type") == "camera_keys":
                    return True
            elif isinstance(value, str):
                candidates.append(value)
            return any(
                token in candidate
                for candidate in candidates
                for token in ("img", "image", "images", "cam", "camera")
            )

        km = {k: v for k, v in km.items() if not _looks_like_image_key(k, v)}

        # Also remove image keys by name heuristic (they won't be in key_map values
        # but the keys themselves may reference image columns we want to skip)
        if OmegaConf.select(instantiate_copy, "resolver.key_map") is not None:
            instantiate_copy.resolver.key_map = km
        else:
            instantiate_copy.key_map = OmegaConf.create(km)

        norm_dataset = hydra.utils.instantiate(instantiate_copy)
        data_schematic.infer_norm_from_dataset(
            norm_dataset,
            dataset_name,
            sample_frac=cfg.norm_stat_fraction,
            max_samples=cfg.get("norm_stat_max_samples"),
            benchmark_dir=os.path.join(
                cfg.trainer.default_root_dir, "benchmark_stats.json"
            ),
        )

    # NOTE: We also pass the data_schematic_dict into the robomimic model's instatiation now that we've initialzied the shapes and norm stats.  In theory, upon loading the PL checkpoint, it will remember this, but let's see.
    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(
        cfg.model, robomimic_model={"data_schematic": data_schematic}
    )

    _log_dataset_frame_counts(train_datasets, valid_datasets)

    log.info("Instantiating callbacks...")
    callbacks: List[Callback] = instantiate_callbacks(cfg.get("callbacks"))

    log.info("Instantiating loggers...")
    logger: List[Logger] = instantiate_loggers(cfg.get("logger"))

    if _env_value_is_enabled("EGOVERSE_AUTO_EXCLUDE_ACTION_MAX_ABS"):
        OmegaConf.update(
            cfg,
            "trainer.reload_dataloaders_every_n_epochs",
            1,
            force_add=True,
        )
        log.info(
            "Enabled trainer.reload_dataloaders_every_n_epochs=1 because "
            "EGOVERSE_AUTO_EXCLUDE_ACTION_MAX_ABS is set."
        )

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    plugins = []
    if os.environ.get("SLURM_JOB_ID"):
        plugins.append(
            SLURMEnvironment(requeue_signal=[signal.SIGUSR1, signal.SIGUSR2])
        )
        print("SLURM REQUEUE ENABLED")
    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer, callbacks=callbacks, logger=logger
    )

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        log_hyperparameters(object_dict)

    if (
        os.environ.get("SLURM_JOB_ID")
        and os.environ.get("SLURM_RESTART_COUNT", "0") != "0"
    ):
        last_ckpt_path = os.path.join(
            trainer.default_root_dir, "checkpoints", "last.ckpt"
        )
        log.info("Detected SLURM requeue — resuming from 'last.ckpt'")
        cfg.ckpt_path = last_ckpt_path

    os.makedirs(os.path.join(trainer.default_root_dir, "videos"), exist_ok=True)

    if cfg.get("train"):
        log.info("Starting training!")
        fit_kwargs = {
            "model": model,
            "datamodule": datamodule,
            "ckpt_path": cfg.get("ckpt_path"),
        }
        if cfg.get("ckpt_path"):
            weights_only = os.environ.get(
                "EGOVERSE_RESUME_WEIGHTS_ONLY", "0"
            ).lower() in {"1", "true", "yes"}
            fit_kwargs["weights_only"] = weights_only
            log.info(
                "Resuming from checkpoint with "
                f"weights_only={weights_only}. "
                "Use EGOVERSE_RESUME_WEIGHTS_ONLY=1 for weights-only loading."
            )
        trainer.fit(**fit_kwargs)

    if cfg.get("eval"):
        eval: Eval = hydra.utils.instantiate(
            cfg.eval_class, config=cfg.model, ckpt_path=cfg.get("ckpt_path")
        )
        log.info("Starting evaluation!")
        eval.perfom_eval()

    train_metrics = trainer.callback_metrics

    # if cfg.get("test"):
    #     log.info("Starting testing!")
    #     ckpt_path = trainer.checkpoint_callback.best_model_path
    #     if ckpt_path == "":
    #         log.warning("Best ckpt not found! Using current weights for testing...")
    #         ckpt_path = None
    #     trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
    #     log.info(f"Best ckpt path: {ckpt_path}")

    # test_metrics = trainer.callback_metrics

    # merge train and test metrics
    test_metrics = {}  # my stub
    metric_dict = {**train_metrics, **test_metrics}

    return metric_dict, object_dict


@hydra.main(
    version_base="1.3", config_path="./hydra_configs", config_name="train_zarr.yaml"
)
def main(cfg: DictConfig) -> Optional[float]:
    """Main entry point for training.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Optional[float] with optimized metric value.
    """
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    extras(cfg)

    print(OmegaConf.to_yaml(cfg))

    # cfg = OmegaConf.resolve(cfg)

    # train the model
    metric_dict, _ = train(cfg)

    # # safely retrieve metric value for hydra-based hyperparameter optimization
    # metric_value = get_metric_value(
    #     metric_dict=metric_dict, metric_name=cfg.get("optimized_metric")
    # )

    # # return optimized metric
    # return metric_value


if __name__ == "__main__":
    main()
