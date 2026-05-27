"""
Freeze the exact Scale episode list used for training into a JSON manifest.

This lets later runs reuse the same episode pool even if the remote SQL table
changes, which keeps experiments comparable and avoids cache growth from pool drift.
"""

import argparse
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from egomimic.rldb.zarr.zarr_dataset_multi import S3EpisodeResolver


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _default_data_config() -> Path:
    return (
        _repo_root()
        / "egomimic"
        / "hydra_configs"
        / "data"
        / "cotrain_viperx_scale.yaml"
    )


def _load_scale_cfg(data_config: Path, split: str) -> tuple[dict, dict]:
    with data_config.open() as f:
        cfg = yaml.safe_load(f)

    dataset_key = "train_datasets" if split == "train" else "valid_datasets"
    scale_cfg = cfg[dataset_key]["scale_bimanual"]
    resolver_cfg = scale_cfg["resolver"]
    filters = scale_cfg.get("filters", {})
    return resolver_cfg, filters


def main():
    parser = argparse.ArgumentParser(
        description="Freeze the exact Scale episode selection into a manifest JSON."
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output JSON manifest path.",
    )
    parser.add_argument(
        "--data-config",
        type=Path,
        default=_default_data_config(),
        help="Hydra data config to read default Scale resolver settings from.",
    )
    parser.add_argument(
        "--split",
        choices=("train", "valid"),
        default="train",
        help="Which Scale resolver block to freeze.",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="Optional override for max_episodes.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Limit the resolver to its debug subset before writing the manifest.",
    )
    args = parser.parse_args()

    resolver_cfg, filters = _load_scale_cfg(args.data_config, args.split)
    max_episodes = (
        args.max_episodes
        if args.max_episodes is not None
        else resolver_cfg.get("max_episodes")
    )

    entries = S3EpisodeResolver._get_filtered_paths(
        filters=filters,
        debug=args.debug,
        max_episodes=max_episodes,
        shuffle_episodes=resolver_cfg.get("shuffle_episodes", False),
        episode_seed=resolver_cfg.get("episode_seed", 42),
        exclude_episode_hashes=resolver_cfg.get("exclude_episode_hashes", []),
    )

    payload = {
        "source_data_config": str(args.data_config),
        "split": args.split,
        "filters": filters,
        "max_episodes": max_episodes,
        "shuffle_episodes": resolver_cfg.get("shuffle_episodes", False),
        "episode_seed": resolver_cfg.get("episode_seed", 42),
        "exclude_episode_hashes": resolver_cfg.get("exclude_episode_hashes", []),
        "episodes": [
            {"processed_path": processed_path, "episode_hash": episode_hash}
            for processed_path, episode_hash in entries
        ],
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        json.dump(payload, f, indent=2)

    print(
        f"Wrote frozen Scale manifest with {len(entries)} episodes to {args.out}",
        flush=True,
    )


if __name__ == "__main__":
    main()
