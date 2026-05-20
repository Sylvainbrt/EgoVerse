import os
import random
from collections import OrderedDict, deque
from typing import Any, Dict

import numpy as np
import torch
import torchvision.io as tvio
from lightning import LightningModule

import egomimic.utils.tensor_utils as TensorUtils
from egomimic.rldb.embodiment.embodiment import get_embodiment


class ModelWrapper(LightningModule):
    """
    Wrapper class around robomimic models to ensure compatibility with Pytorch Lightning.
    """

    debug_loss_spike = False
    debug_loss_spike_factor = 1000.0
    debug_loss_spike_prob = 0.03
    grad_norm_mad_scale = 3.0
    grad_norm_mad_min_count = 100
    grad_norm_mad_window = 200

    def __init__(self, robomimic_model, optimizer, scheduler):
        """
        Args:
            model (PolicyAlgo): robomimic model to wrap.
        """
        super().__init__()
        self.save_hyperparameters()

        self.model = robomimic_model
        self.nets = (
            self.model.nets
        )  # to ensure the lightning module has access to the model's parameters
        try:
            self.params = self.model.nets["policy"].params
        except Exception:
            pass
        self.grad_norm_history = deque(maxlen=self.grad_norm_mad_window)

        self.val_image_buffer, self.val_counter = {}, {}
        self.epoch_memory_stats = []  # Store memory stats per epoch
        # TODO __init__ should take the config, and init the model here.  Then save_hyperparameters will just save the config rather than the model

    def root_dir(self):
        return self.trainer.default_root_dir

    def video_dir(self):
        return os.path.join(self.root_dir(), "videos")

    def _debug_spike_threshold(self):
        value = os.environ.get("EGOVERSE_DEBUG_LOSS_THRESHOLD")
        if value is None:
            return None
        try:
            return float(value)
        except ValueError:
            if self.trainer.is_global_zero:
                print(
                    f"[LOSS_SPIKE_DEBUG] invalid EGOVERSE_DEBUG_LOSS_THRESHOLD={value!r}",
                    flush=True,
                )
            return None

    def _summarize_batch_sources(self, raw_batch):
        summary = {}
        for embodiment_name, domain_batch in raw_batch.items():
            domain_summary = {}
            for key in (
                "metadata.episode_hash",
                "metadata.frame_idx",
                "episode_index",
                "frame_index",
                "index",
            ):
                if key not in domain_batch:
                    continue
                value = domain_batch[key]
                if isinstance(value, torch.Tensor):
                    value = value.detach().cpu().tolist()
                elif isinstance(value, tuple):
                    value = list(value)
                domain_summary[key] = value
            for key in ("actions_cartesian", "actions_joints", "action"):
                if key not in domain_batch or not isinstance(
                    domain_batch[key], torch.Tensor
                ):
                    continue
                value = domain_batch[key].detach().float().cpu()
                flat = value.reshape(value.shape[0], -1)
                finite = torch.isfinite(flat).all(dim=1)
                max_abs = flat.abs().amax(dim=1)
                rms = torch.sqrt(torch.mean(flat.square(), dim=1))
                k = min(5, len(max_abs))
                top = torch.topk(max_abs, k=k).indices.tolist()
                domain_summary[f"{key}.shape"] = list(value.shape)
                domain_summary[f"{key}.top_max_abs"] = [
                    {
                        "batch_i": int(i),
                        "max_abs": float(max_abs[i]),
                        "rms": float(rms[i]),
                        "finite": bool(finite[i]),
                    }
                    for i in top
                ]
            if domain_summary:
                summary[embodiment_name] = domain_summary
        return summary

    def _maybe_log_loss_spike_sources(self, raw_batch, losses):
        threshold = self._debug_spike_threshold()
        if threshold is None or not self.trainer.is_global_zero:
            return

        spike_losses = {}
        for key, value in losses.items():
            if not key.endswith("_loss"):
                continue
            scalar = float(value.detach().cpu())
            if scalar >= threshold:
                spike_losses[key] = scalar

        if not spike_losses:
            return

        print(
            "[LOSS_SPIKE_DEBUG] "
            f"global_step={self.global_step} "
            f"epoch={self.current_epoch} "
            f"losses={spike_losses} "
            f"sources={self._summarize_batch_sources(raw_batch)}",
            flush=True,
        )

    # batch is now a dict, handle on model side
    def training_step(self, batch, batch_idx):
        self.train()
        raw_batch = batch
        loss_dicts = []
        batch = self.model.process_batch_for_training(batch)
        predictions = self.model.forward_training(batch)
        losses = self.model.compute_losses(predictions, batch)
        loss_dicts.append(losses)

        # Average over both the hand and robot batch if applicable
        losses = OrderedDict()
        for key in loss_dicts[0].keys():
            losses[key] = torch.mean(
                torch.stack([loss_dict[key] for loss_dict in loss_dicts])
            )

        if (
            self.debug_loss_spike
            and random.random() < self.debug_loss_spike_prob
            and self.global_step > 100
        ):
            losses["action_loss"] = losses["action_loss"] * self.debug_loss_spike_factor
            if self.trainer.is_global_zero:
                print(
                    "[LOSS_SPIKE] "
                    f"step={self.global_step} "
                    f"factor={self.debug_loss_spike_factor}",
                    flush=True,
                )

        self._maybe_log_loss_spike_sources(raw_batch, losses)

        info = {}
        info["losses"] = TensorUtils.detach(losses)
        for k, v in self.model.log_info(info).items():
            self.log("Train/" + k, v, sync_dist=True, on_step=False, on_epoch=True)

        return losses["action_loss"]

    def on_after_backward(self):
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.parameters(), max_norm=float("inf")
        )
        grad_norm_val = float(grad_norm)
        info = {"policy_grad_norms_raw": grad_norm_val}

        if len(self.grad_norm_history) >= self.grad_norm_mad_min_count:
            values = np.array(self.grad_norm_history, dtype=np.float32)
            median = float(np.median(values))
            mad = float(np.median(np.abs(values - median)))
            if mad > 0.0:
                threshold = median + self.grad_norm_mad_scale * mad
                info["policy_grad_norms_mad_threshold"] = threshold
                info["policy_grad_norms_mad_flag"] = float(grad_norm_val > threshold)
                if grad_norm_val > threshold:
                    torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=median)
                    if self.trainer.is_global_zero:
                        print(
                            "[GRAD_NORM_SPIKE] "
                            f"step={self.global_step} "
                            f"grad_norm={grad_norm_val:.4f} "
                            f"median={median:.4f} "
                            f"mad={mad:.4f} "
                            f"threshold={threshold:.4f}",
                            flush=True,
                        )

        self.grad_norm_history.append(grad_norm_val)
        for k, v in info.items():
            self.log("Train/" + k, v, on_step=False, on_epoch=True, sync_dist=True)

    def on_before_optimizer_step(self, optimizer):
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.parameters(), max_norm=float("inf")
        )
        self.log(
            "Train/policy_grad_norms_clipped",
            float(grad_norm),
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )

    def on_validation_start(self):
        self.model.device = self.device

        if self.trainer.is_global_zero:
            os.makedirs(
                os.path.join(self.video_dir(), f"epoch_{self.trainer.current_epoch}"),
                exist_ok=True,
            )

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        """
        Run a validation step on the batch, and save that batch of images into the val_image_buffer.  Once the buffer hits 1000 images, save that as a 30fps video using torchvision.io.write_video.
        """
        if batch_idx < 5 or batch_idx % 50 == 0:
            print(
                f"[VAL_STEP] rank={self.global_rank}, batch_idx={batch_idx}",
                flush=True,
            )

        batch = self.model.process_batch_for_training(batch)
        metrics, images_dict = self.model.forward_eval_logging(batch)

        metrics = {
            k: (
                v.to(self.device)
                if torch.is_tensor(v)
                else torch.tensor(v, device=self.device)
            )
            for k, v in metrics.items()
        }

        ## images is now a dict
        for key, images in images_dict.items():
            os.makedirs(
                os.path.join(
                    self.video_dir(),
                    f"epoch_{self.trainer.current_epoch}",
                    str(get_embodiment(key)),
                ),
                exist_ok=True,
            )
            if key not in self.val_image_buffer or self.val_image_buffer[key] is None:
                self.val_image_buffer[key] = []
                self.val_counter[key] = 0
            self.val_image_buffer[key].extend(torch.from_numpy(images))
            if len(self.val_image_buffer[key]) >= 1000:
                frames = torch.stack(self.val_image_buffer[key])
                path = os.path.join(
                    self.video_dir(),
                    f"epoch_{self.trainer.current_epoch}",
                    str(get_embodiment(key)),
                    f"validation_video_{self.val_counter[key]}.mp4",
                )
                tvio.write_video(path, frames, fps=30, video_codec="h264")
                self.val_image_buffer[key].clear()
                self.val_counter[key] += 1

        self.log_dict(metrics, sync_dist=True)

    def on_validation_end(self):
        print(f"[ON_VALIDATION_END] rank={self.global_rank}", flush=True)
        for key, buffer in self.val_image_buffer.items():
            os.makedirs(
                os.path.join(
                    self.video_dir(),
                    f"epoch_{self.trainer.current_epoch}",
                    str(get_embodiment(key)),
                ),
                exist_ok=True,
            )
            if len(buffer) != 0:
                frames = torch.stack(buffer)
                path = os.path.join(
                    self.video_dir(),
                    f"epoch_{self.trainer.current_epoch}",
                    str(get_embodiment(key)),
                    f"validation_video_{self.val_counter[key]}.mp4",
                )
                tvio.write_video(path, frames, fps=30, video_codec="h264")

            self.val_counter[key] = 0
            self.val_image_buffer[key] = []
        print(
            f"Rank {self.global_rank} on validation end, waiting for all ranks to synchronize",
            flush=True,
        )
        torch.distributed.barrier()
        print(
            f"Rank {self.global_rank} on validation end, all ranks synchronized",
            flush=True,
        )

    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": scheduler,
                # "lr_scheduler": {
                #     "scheduler": scheduler,
                #     "monitor": "val/loss",
                #     "interval": "epoch",
                #     "frequency": 1,
                # },
            }
        return {"optimizer": optimizer}

    def on_fit_start(self):
        self.model.device = self.device
        print(
            f"Rank {self.global_rank} on fit start, waiting for all ranks to synchronize",
            flush=True,
        )
        torch.distributed.barrier()
        print(
            f"Rank {self.global_rank} on fit start, all ranks synchronized", flush=True
        )

    def on_train_epoch_start(self):
        log_all = {}
        for i, param_group in enumerate(self.optimizers().param_groups):
            log_all[f"Optimizer/param_group_{i}_lr"] = param_group["lr"]

        return super().on_train_epoch_start()
