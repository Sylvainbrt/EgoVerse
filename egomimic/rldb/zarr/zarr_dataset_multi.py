"""
ZarrDataset implementation for EgoVerse.

Mirrors the LeRobotDataset API while reading data from Zarr arrays
instead of parquet/HF datasets.

Directory structure (per-episode metadata):
    dataset_root/
    └── episode_{ep_idx}.zarr/
        ├── observations.images.{cam}  (JPEG compressed)
        ├── observations.state
        ├── actions_joints
        └── ...

Each episode is self-contained with its own metadata, enabling:
- Independent episode uploads to S3
- Parallel processing without global coordination
- Easy episode-level data management
"""

from __future__ import annotations

import json
import logging
import os
import random
import subprocess
import tempfile
from pathlib import Path
from typing import Iterable
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import simplejpeg
import torch
import zarr

# from action_chunk_transforms import Transform
from egomimic.utils.aws.aws_data_utils import load_env
from egomimic.utils.aws.aws_sql import (
    create_default_engine,
    episode_table_to_df,
)

logger = logging.getLogger(__name__)

SEED = 42


def split_dataset_names(dataset_names, valid_ratio=0.2, seed=SEED):
    """
    Split a list of dataset names into train/valid sets.
    Args:
        dataset_names (Iterable[str])
        valid_ratio (float): fraction of datasets to put in valid.
        seed (int): for deterministic shuffling.


    Returns:
        train_set (set[str]), valid_set (set[str])
    """
    names = sorted(dataset_names)
    if not names:
        return set(), set()

    rng = random.Random(seed)
    rng.shuffle(names)

    if not (0.0 <= valid_ratio <= 1.0):
        raise ValueError(f"valid_ratio must be in [0,1], got {valid_ratio}")

    n_valid = int(len(names) * valid_ratio)
    if valid_ratio > 0.0:
        n_valid = max(1, n_valid)

    valid = set(names[:n_valid])
    train = set(names[n_valid:])
    return train, valid


class EpisodeResolver:
    """
    Base class for episode resolution utilities.
    Provides shared static/class helpers; subclasses implement resolve().
    """

    def __init__(
        self,
        folder_path: Path,
        key_map: dict | None = None,
        transform_list: list | None = None,
    ):
        self.folder_path = Path(folder_path)
        self.key_map = key_map
        self.transform_list = transform_list

    def _build_zarr_datasets(
        self, episode_paths: list[tuple[str | Path, str]]
    ) -> dict[str, "ZarrDataset"]:
        """
        Build ZarrDataset objects from explicit (episode_path, episode_hash) tuples.

        Args:
            episode_paths: List of tuples containing the episode path/URI and hash.

        Returns:
            dict[str, ZarrDataset]: Mapping from episode hash to dataset object.
        """
        from tqdm import tqdm

        datasets: dict[str, ZarrDataset] = {}

        for episode_path, episode_hash in tqdm(
            episode_paths,
            desc="Loading datasets",
            unit="episode",
        ):
            try:
                ds_obj = ZarrDataset(
                    episode_path,
                    key_map=self.key_map,
                    transform_list=self.transform_list,
                )
                datasets[episode_hash] = ds_obj

            except Exception as e:
                logger.error(
                    "Failed to load dataset for episode %s at %s: %s",
                    episode_hash,
                    episode_path,
                    e,
                )

        return datasets

    def _load_zarr_datasets(self, search_path: Path, valid_folder_names: set[str]):
        """
        Loads multiple Zarr datasets from the specified folder path, filtering only those whose hashes
        are present in the valid_folder_names set.

        Args:
            search_path (Path): The root directory to search for Zarr datasets.
            valid_folder_names (set[str]): A set of valid folder names (episode hashes without ".zarr") to filter datasets.
        Returns:
            dict[str, ZarrDataset]: a dictionary mapping string keys to constructed zarr datasets from valid filters.
        """
        all_paths = sorted(search_path.iterdir())
        skipped: list[str] = []
        filtered_paths: list[tuple[str | Path, str]] = []
        for p in all_paths:
            if not p.is_dir():
                logger.info(f"{p} is not a valid directory")
                skipped.append(p.name)
                continue
            name = p.name
            if name.endswith(".zarr"):
                name = name[: -len(".zarr")]
            if name not in valid_folder_names:
                skipped.append(p.name)
                continue
            filtered_paths.append((p, name))

        return self._build_zarr_datasets(filtered_paths)

    @classmethod
    def _episode_already_present(cls, local_dir: Path, episode_hash: str) -> bool:
        direct = local_dir / episode_hash
        if direct.is_dir() and (direct / "zarr.json").is_file():
            return True
        zarr_dir = local_dir / f"{episode_hash}.zarr"
        if zarr_dir.is_dir() and (zarr_dir / "zarr.json").is_file():
            return True
        return False

    @classmethod
    def _local_episode_path(cls, local_dir: Path, episode_hash: str) -> Path:
        direct = local_dir / episode_hash
        if direct.is_dir():
            return direct
        zarr_dir = local_dir / f"{episode_hash}.zarr"
        if zarr_dir.is_dir():
            return zarr_dir
        return direct


class S3EpisodeResolver(EpisodeResolver):
    """
    Resolves episodes via SQL table and optionally syncs from S3.
    """

    _known_unavailable_remote_episodes: set[tuple[str, str, str]] = set()

    def __init__(
        self,
        folder_path: Path,
        bucket_name: str = "rldb",
        main_prefix: str = "processed_v3",
        key_map: dict | None = None,
        transform_list: list | None = None,
        debug: bool = False,
        max_episodes: int | None = None,
        shuffle_episodes: bool = False,
        episode_seed: int | None = 42,
        exclude_episode_hashes: list[str] | None = None,
    ):
        self.bucket_name = bucket_name
        self.main_prefix = main_prefix
        self.debug = debug
        self.max_episodes = max_episodes
        self.shuffle_episodes = shuffle_episodes
        self.episode_seed = episode_seed
        self.exclude_episode_hashes = exclude_episode_hashes or []
        super().__init__(folder_path, key_map=key_map, transform_list=transform_list)

    @classmethod
    def _remote_episode_key(
        cls, bucket_name: str, processed_path: str, episode_hash: str
    ) -> tuple[str, str, str]:
        return (
            bucket_name,
            cls._normalize_s3_uri(bucket_name, str(processed_path)),
            str(episode_hash),
        )

    @classmethod
    def _is_remote_episode_known_unavailable(
        cls, bucket_name: str, processed_path: str, episode_hash: str
    ) -> bool:
        return (
            cls._remote_episode_key(bucket_name, processed_path, episode_hash)
            in cls._known_unavailable_remote_episodes
        )

    @classmethod
    def _mark_remote_episodes_unavailable(
        cls, bucket_name: str, s3_paths: list[tuple[str, str]]
    ) -> None:
        for processed_path, episode_hash in s3_paths:
            cls._known_unavailable_remote_episodes.add(
                cls._remote_episode_key(bucket_name, processed_path, episode_hash)
            )

    def resolve(
        self,
        filters: dict | None = None,
    ) -> dict[str, "ZarrDataset"]:
        """
        Outputs a dict of ZarrDatasets with relevant filters.
        Syncs S3 paths to local_root before indexing.
        """
        filters = dict(filters) if filters is not None else {}
        filters["is_deleted"] = False

        if self.folder_path.is_dir():
            logger.info(f"Using existing directory: {self.folder_path}")
        if not self.folder_path.is_dir():
            self.folder_path.mkdir()

        logger.info(f"Filters: {filters}")

        filtered_paths = self.sync_from_filters(
            bucket_name=self.bucket_name,
            filters=filters,
            local_dir=self.folder_path,
            debug=self.debug,
            max_episodes=self.max_episodes,
            shuffle_episodes=self.shuffle_episodes,
            episode_seed=self.episode_seed,
            exclude_episode_hashes=self.exclude_episode_hashes,
        )

        valid_hashes = {hashes for _, hashes in filtered_paths}
        if not valid_hashes:
            raise ValueError(
                "No valid collection names from _get_filtered_paths: "
                "filters matched no episodes in the SQL table."
            )

        local_episode_paths = [
            (self._local_episode_path(self.folder_path, episode_hash), episode_hash)
            for _, episode_hash in filtered_paths
        ]
        datasets = self._build_zarr_datasets(
            episode_paths=local_episode_paths,
        )

        return datasets

    @staticmethod
    def _normalize_s3_uri(bucket_name: str, processed_path: str) -> str:
        if processed_path.startswith("s3://"):
            return processed_path.rstrip("/")
        return f"s3://{bucket_name}/{processed_path.lstrip('/').rstrip('/')}"

    @staticmethod
    def _get_filtered_paths(
        filters: dict | None = None,
        debug: bool = False,
        max_episodes: int | None = None,
        shuffle_episodes: bool = False,
        episode_seed: int | None = 42,
        exclude_episode_hashes: list[str] | None = None,
    ) -> list[tuple[str, str]]:
        """
        Filters episodes from the SQL episode table according to the criteria specified in `filters`
        and returns a list of (zarr_processed_path, episode_hash) tuples for episodes that match and
        have a non-null zarr_processed_path.

        Args:
            filters (dict): Dictionary of filter key-value pairs to apply on the episode table.

        Returns:
            list[tuple[str, str]]: List of tuples, each containing (zarr_processed_path, episode_hash)
                                   for episodes passing the filter criteria.
        """
        filters = dict(filters) if filters is not None else {}
        engine = create_default_engine()
        df = episode_table_to_df(engine)
        if df.empty:
            logger.warning("Episode table is empty after SQL query.")
            return []

        mask = pd.Series(True, index=df.index)
        for key, value in filters.items():
            if key == "robot_name":
                candidate_cols = [
                    col
                    for col in ("robot_name", "robot_type", "embodiment")
                    if col in df.columns
                ]
                if not candidate_cols:
                    raise KeyError(
                        "Could not apply filter 'robot_name': none of "
                        "'robot_name', 'robot_type', or 'embodiment' exist in the "
                        f"episode table columns {list(df.columns)}"
                    )
                column_mask = pd.Series(False, index=df.index)
                for col in candidate_cols:
                    column_mask |= df[col] == value
                mask &= column_mask
            elif key == "is_deleted" and key not in df.columns:
                mask &= bool(value) is False
            else:
                if key not in df.columns:
                    raise KeyError(
                        f"Could not apply filter '{key}': column not found in "
                        f"episode table columns {list(df.columns)}"
                    )
                mask &= df[key] == value

        output = df.loc[mask, ["zarr_processed_path", "episode_hash"]]
        before_len = len(output)

        if debug:
            logger.info("Debug mode: limiting to 10 datasets.")
            output = output.head(10)

        output = output[
            output["zarr_processed_path"].fillna("").astype(str).str.strip() != ""
        ]
        logger.info(
            f"Skipped {before_len - len(output)} episodes with null zarr_processed_path: {output}"
        )

        if exclude_episode_hashes:
            before_exclude = len(output)
            output = output[~output["episode_hash"].isin(exclude_episode_hashes)]
            logger.info(
                "Excluded %d episode hashes from resolver",
                before_exclude - len(output),
            )

        if shuffle_episodes:
            output = output.sample(frac=1.0, random_state=episode_seed).reset_index(
                drop=True
            )
            logger.info("Shuffled episodes with seed=%s", episode_seed)

        if max_episodes is not None and max_episodes > 0:
            output = output.head(max_episodes)
            logger.info(f"Limited output to {max_episodes} episodes")

        paths = list(output.itertuples(index=False, name=None))
        logger.info(f"Paths: {paths}")
        return paths

    @classmethod
    def _sync_s3_to_local(
        cls,
        bucket_name: str,
        s3_paths: list[tuple[str, str]],
        local_dir: Path,
        numworkers: int = 10,
    ):
        if not s3_paths:
            return

        # 0) Skip episodes already present locally
        to_sync = []
        already = []
        known_unavailable = []
        for processed_path, episode_hash in s3_paths:
            if cls._is_remote_episode_known_unavailable(
                bucket_name, processed_path, episode_hash
            ):
                known_unavailable.append(episode_hash)
            elif cls._episode_already_present(local_dir, episode_hash):
                already.append(episode_hash)
            else:
                to_sync.append((processed_path, episode_hash))

        if already:
            logger.info("Skipping %d episodes already present locally.", len(already))
        if known_unavailable:
            logger.info(
                "Skipping %d episodes previously marked unavailable on remote.",
                len(known_unavailable),
            )

        if not to_sync:
            logger.info("Nothing to sync from S3 (all episodes already present).")
            return

        # 1) Build s5cmd batch script (one line per episode)
        local_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            prefix="_s5cmd_sync_",
            suffix=".txt",
            delete=False,
        ) as tmp_file:
            batch_path = Path(tmp_file.name)

        lines = []
        for processed_path, episode_hash in to_sync:
            # processed_path like: s3://rldb/processed_v2/eva/<hash>/
            if processed_path.startswith("s3://"):
                src_prefix = processed_path.rstrip("/") + "/*"
            else:
                src_prefix = (
                    f"s3://{bucket_name}/{processed_path.lstrip('/').rstrip('/')}"
                    + "/*"
                )

            # Destination is the root local_dir; s5cmd will preserve <hash>/... under it
            dst = local_dir / episode_hash
            lines.append(f'sync "{src_prefix}" "{str(dst)}/"')

        try:
            batch_path.write_text("\n".join(lines) + "\n")

            load_env()
            rl2_endpoint_url = os.environ.get("R2_ENDPOINT_URL")
            access_key_id = os.environ.get("R2_ACCESS_KEY_ID")
            secret_access_key = os.environ.get("R2_SECRET_ACCESS_KEY")
            if not all([rl2_endpoint_url, access_key_id, secret_access_key]):
                raise ValueError(
                    "R2 credentials missing. Ensure ~/.egoverse_env has "
                    "R2_ENDPOINT_URL, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY."
                )
            s5cmd_env = os.environ.copy()
            s5cmd_env["AWS_ACCESS_KEY_ID"] = access_key_id
            s5cmd_env["AWS_SECRET_ACCESS_KEY"] = secret_access_key
            s5cmd_env["AWS_DEFAULT_REGION"] = "auto"
            s5cmd_env["AWS_REGION"] = "auto"
            cmd = [
                "s5cmd",
                "--endpoint-url",
                rl2_endpoint_url,
                "--numworkers",
                str(numworkers),
                "run",
                str(batch_path),
            ]
            logger.info("Running s5cmd batch (%d lines): %s", len(lines), " ".join(cmd))
            result = subprocess.run(
                cmd,
                check=False,
                env=s5cmd_env,
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                combined_output = "\n".join(
                    chunk for chunk in [result.stdout, result.stderr] if chunk
                )
                error_lines = [
                    line.strip()
                    for line in combined_output.splitlines()
                    if line.strip() and "error" in line.lower()
                ]
                tolerated_missing_markers = [
                    "no object found",
                    "object not found",
                    "key does not exist",
                ]
                only_missing_object_errors = bool(error_lines) and all(
                    any(marker in line.lower() for marker in tolerated_missing_markers)
                    for line in error_lines
                )
                if only_missing_object_errors:
                    logger.warning(
                        "s5cmd reported %d missing remote objects while syncing; "
                        "continuing and letting the resolver drop unavailable "
                        "episodes. First errors: %s",
                        len(error_lines),
                        error_lines[:5],
                    )
                else:
                    raise subprocess.CalledProcessError(
                        result.returncode,
                        cmd,
                        output=result.stdout,
                        stderr=result.stderr,
                    )

        finally:
            try:
                batch_path.unlink(missing_ok=True)
            except Exception as e:
                logger.warning("Failed to delete batch file %s: %s", batch_path, e)

    @classmethod
    def sync_from_filters(
        cls,
        *,
        bucket_name: str,
        filters: dict,
        local_dir: Path,
        numworkers: int = 10,
        debug: bool = False,
        max_episodes=None,
        shuffle_episodes: bool = False,
        episode_seed: int | None = 42,
        exclude_episode_hashes: list[str] | None = None,
    ):
        """
        Public API:
        - resolves episodes from DB using filters
        - runs a single aws s3 sync with includes
        - downloads into local_dir

        Args:
            numworkers: Number of parallel workers for s5cmd.

        Returns:
            List[(processed_path, episode_hash)]
        """

        # 1) Resolve episodes from DB
        filtered_paths = cls._get_filtered_paths(
            filters,
            debug=debug,
            max_episodes=max_episodes,
            shuffle_episodes=shuffle_episodes,
            episode_seed=episode_seed,
            exclude_episode_hashes=exclude_episode_hashes,
        )
        if not filtered_paths:
            logger.warning("No episodes matched filters.")
            return []

        # 2) Logging
        logger.info(
            f"Syncing S3 datasets with filters {filters} to local directory {local_dir}..."
        )

        # 3) Sync
        cls._sync_s3_to_local(
            bucket_name=bucket_name,
            s3_paths=filtered_paths,
            local_dir=local_dir,
            numworkers=numworkers,
        )

        return filtered_paths


class LocalEpisodeResolver(EpisodeResolver):
    """
    Resolves episodes from local Zarr stores, filtering via local metadata.
    """

    def __init__(
        self,
        folder_path: Path,
        key_map: dict | None = None,
        transform_list: list | None = None,
        debug=False,
    ):
        super().__init__(folder_path, key_map, transform_list)
        self.debug = debug

    @staticmethod
    def _local_filters_match(metadata: dict, episode_hash: str, filters: dict) -> bool:
        for key, value in filters.items():
            if key == "episode_hash":
                if episode_hash != value:
                    return False
                continue

            if key == "robot_name":
                meta_value = (
                    metadata.get("robot_name")
                    or metadata.get("robot_type")
                    or metadata.get("embodiment")
                )
            elif key == "is_deleted":
                meta_value = metadata.get("is_deleted", False)
            else:
                meta_value = metadata.get(key)

            if meta_value is None:
                return False
            if meta_value != value:
                return False

        return True

    @classmethod
    def _get_local_filtered_paths(
        cls, search_path: Path, filters: dict, debug: bool = False
    ):
        if not search_path.is_dir():
            logger.warning("Local path does not exist: %s", search_path)
            return []

        filtered = []
        for p in sorted(search_path.iterdir()):
            if not p.is_dir():
                continue

            episode_hash = p.name[:-5] if p.name.endswith(".zarr") else p.name

            try:
                store = zarr.open_group(str(p), mode="r")
                metadata = dict(store.attrs)
            except Exception as e:
                logger.warning("Failed to read metadata for %s: %s", p, e)
                continue

            if cls._local_filters_match(metadata, episode_hash, filters):
                filtered.append((str(p), episode_hash))

        if debug:
            logger.info("Debug mode: limiting to 10 datasets.")
            filtered = filtered[:10]

        logger.info("Local filtered paths: %s", filtered)
        return filtered

    def resolve(
        self,
        sync_from_s3=False,
        filters: dict | None = None,
    ) -> dict[str, "ZarrDataset"]:
        """
        Outputs a dict of ZarrDatasets with relevant filters from local data.
        """
        if sync_from_s3:
            logger.warning(
                "LocalEpisodeResolver does not sync from S3; ignoring sync_from_s3=True."
            )

        filters = dict(filters) if filters is not None else {}
        filters.setdefault("is_deleted", False)

        filtered_paths = self._get_local_filtered_paths(
            self.folder_path, filters, debug=self.debug
        )

        valid_folder_names = {folder_name for _, folder_name in filtered_paths}
        logger.info(f"Valid folder names: {valid_folder_names}")
        if not valid_folder_names:
            raise ValueError(
                "No valid collection names from local filtering: "
                "filters matched no episodes in the local directory."
            )

        datasets = self._build_zarr_datasets(
            episode_paths=filtered_paths,
        )

        return datasets


class S3StreamingEpisodeResolver(S3EpisodeResolver):
    """
    Resolves episodes from SQL metadata and reads them directly from S3/R2.
    """

    def resolve(
        self,
        filters: dict | None = None,
    ) -> dict[str, "ZarrDataset"]:
        filters = dict(filters) if filters is not None else {}
        filters["is_deleted"] = False

        logger.info("Streaming S3 datasets with filters: %s", filters)
        filtered_paths = self._get_filtered_paths(
            filters,
            debug=self.debug,
            max_episodes=self.max_episodes,
            shuffle_episodes=self.shuffle_episodes,
            episode_seed=self.episode_seed,
            exclude_episode_hashes=self.exclude_episode_hashes,
        )

        if not filtered_paths:
            raise ValueError(
                "No valid collection names from _get_filtered_paths: "
                "filters matched no episodes in the SQL table."
            )

        remote_episode_paths = [
            (self._normalize_s3_uri(self.bucket_name, processed_path), episode_hash)
            for processed_path, episode_hash in filtered_paths
        ]
        return self._build_zarr_datasets(remote_episode_paths)


class ManifestEpisodeResolver(EpisodeResolver):
    """
    Resolves episodes from a frozen manifest file instead of querying SQL.

    The manifest freezes the exact list of episode hashes / processed paths so later
    training runs reuse the same episode pool even if the remote SQL table changes.
    """

    def __init__(
        self,
        folder_path: Path,
        manifest_path: str | Path,
        bucket_name: str = "rldb",
        key_map: dict | None = None,
        transform_list: list | None = None,
        sync_missing: bool = True,
        max_episodes: int | None = None,
        shuffle_episodes: bool = False,
        episode_seed: int | None = 42,
        exclude_episode_hashes: list[str] | None = None,
    ):
        self.manifest_path = Path(manifest_path)
        self.bucket_name = bucket_name
        self.sync_missing = sync_missing
        self.max_episodes = max_episodes
        self.shuffle_episodes = shuffle_episodes
        self.episode_seed = episode_seed
        self.exclude_episode_hashes = exclude_episode_hashes or []
        super().__init__(folder_path, key_map=key_map, transform_list=transform_list)

    def _load_manifest_entries(self) -> list[tuple[str, str]]:
        if not self.manifest_path.is_file():
            raise FileNotFoundError(f"Manifest file not found: {self.manifest_path}")

        with self.manifest_path.open() as f:
            payload = json.load(f)

        episodes = payload.get("episodes", payload)
        if not isinstance(episodes, list):
            raise ValueError(
                f"Manifest must contain a list or an 'episodes' list: {self.manifest_path}"
            )

        entries: list[tuple[str, str]] = []
        for item in episodes:
            if isinstance(item, dict):
                processed_path = item.get("processed_path")
                episode_hash = item.get("episode_hash")
            elif isinstance(item, (list, tuple)) and len(item) == 2:
                processed_path, episode_hash = item
            else:
                raise ValueError(
                    f"Unsupported manifest episode entry {item!r} in {self.manifest_path}"
                )

            if not processed_path or not episode_hash:
                raise ValueError(
                    f"Manifest episode entry missing processed_path/episode_hash: {item!r}"
                )

            entries.append((str(processed_path), str(episode_hash)))

        return entries

    def _apply_manifest_filters(
        self, entries: list[tuple[str, str]]
    ) -> list[tuple[str, str]]:
        if not self.exclude_episode_hashes:
            return entries

        excluded = set(self.exclude_episode_hashes)
        filtered = [
            (processed_path, episode_hash)
            for processed_path, episode_hash in entries
            if episode_hash not in excluded
        ]
        if len(filtered) != len(entries):
            logger.info(
                "Excluded %d episode hashes from frozen manifest",
                len(entries) - len(filtered),
            )
        return filtered

    def _resolve_local_episode_paths(
        self, entries: list[tuple[str, str]]
    ) -> tuple[list[tuple[Path, str]], list[str]]:
        local_episode_paths: list[tuple[Path, str]] = []
        missing: list[str] = []
        for _, episode_hash in entries:
            local_path = self._local_episode_path(self.folder_path, episode_hash)
            if not local_path.is_dir():
                missing.append(episode_hash)
                continue
            local_episode_paths.append((local_path, episode_hash))
        return local_episode_paths, missing

    def resolve(
        self,
        filters: dict | None = None,
    ) -> dict[str, "ZarrDataset"]:
        if filters:
            logger.warning(
                "ManifestEpisodeResolver ignores runtime filters and uses the frozen manifest: %s",
                self.manifest_path,
            )

        entries = self._apply_manifest_filters(self._load_manifest_entries())
        logger.info(
            "Loaded %d frozen manifest episodes from %s",
            len(entries),
            self.manifest_path,
        )
        if not entries:
            raise ValueError(
                f"No manifest episodes available after exclusions: {self.manifest_path}"
            )

        self.folder_path.mkdir(parents=True, exist_ok=True)

        target_episodes = len(entries)
        if self.max_episodes is not None and self.max_episodes > 0:
            target_episodes = min(self.max_episodes, len(entries))

        if not self.sync_missing:
            local_episode_paths, missing = self._resolve_local_episode_paths(
                entries[:target_episodes]
            )
            if missing:
                raise FileNotFoundError(
                    "Frozen manifest references episodes missing from local cache: "
                    f"{missing[:10]}{'...' if len(missing) > 10 else ''}"
                )
            return self._build_zarr_datasets(local_episode_paths)

        resolved_entries: list[tuple[Path, str]] = []
        unavailable: list[str] = []
        cursor = 0
        while cursor < len(entries) and len(resolved_entries) < target_episodes:
            batch_size = target_episodes - len(resolved_entries)
            batch = entries[cursor : cursor + batch_size]
            cursor += len(batch)
            if not batch:
                break

            S3EpisodeResolver._sync_s3_to_local(
                bucket_name=self.bucket_name,
                s3_paths=batch,
                local_dir=self.folder_path,
            )
            local_episode_paths, missing = self._resolve_local_episode_paths(batch)
            resolved_entries.extend(local_episode_paths)
            unavailable.extend(missing)
            if missing:
                missing_set = set(missing)
                S3EpisodeResolver._mark_remote_episodes_unavailable(
                    self.bucket_name,
                    [
                        (processed_path, episode_hash)
                        for processed_path, episode_hash in batch
                        if episode_hash in missing_set
                    ],
                )

        if len(resolved_entries) < target_episodes and not resolved_entries:
            raise FileNotFoundError(
                "Frozen manifest could only resolve "
                f"{len(resolved_entries)}/{target_episodes} requested episodes after "
                f"syncing from {self.manifest_path}. Unavailable hashes: "
                f"{unavailable[:10]}{'...' if len(unavailable) > 10 else ''}"
            )

        if len(resolved_entries) < target_episodes:
            logger.warning(
                "Frozen manifest could only resolve %d/%d requested episodes from %s; "
                "proceeding with the available subset. First unavailable hashes: %s",
                len(resolved_entries),
                target_episodes,
                self.manifest_path,
                unavailable[:10],
            )
        elif unavailable:
            logger.warning(
                "Skipped %d unavailable frozen manifest episodes while resolving %d "
                "requested episodes from %s. First missing hashes: %s",
                len(unavailable),
                target_episodes,
                self.manifest_path,
                unavailable[:10],
            )

        return self._build_zarr_datasets(resolved_entries)


class MultiDataset(torch.utils.data.Dataset):
    """
    Self wrapping MultiDataset, can wrap zarr or multi dataset.

    """

    def __init__(
        self,
        datasets,
        mode="train",
        percent=0.1,
        valid_ratio=0.2,
        **kwargs,
    ):
        """
        Args:
            datasets (dict): Dictionary mapping unique dataset hashes (str) to dataset objects. Datasets can be individual Zarr datasets or other multi-datasets; mixing different types is supported.
            mode (str, optional): Split mode to use (e.g., "train", "valid"). Defaults to "train".
            percent (float, optional): Fraction of the dataset to use from each underlying dataset. Defaults to 0.1.
            valid_ratio (float, optional): Validation split ratio for datasets that support a train/valid split.
            **kwargs: Additional keyword arguments passed to underlying dataset constructors if needed.
        """
        self.train_collections, self.valid_collections = split_dataset_names(
            datasets.keys(), valid_ratio=valid_ratio, seed=SEED
        )

        if mode == "train":
            chosen = self.train_collections
        elif mode == "valid":
            chosen = self.valid_collections
        elif mode == "total":
            chosen = set(datasets.keys())
        elif mode == "percent":
            all_names = sorted(datasets.keys())
            rng = random.Random(SEED)
            rng.shuffle(all_names)

            n_keep = int(len(all_names) * percent)
            if percent > 0.0:
                n_keep = max(1, n_keep)
            chosen = set(all_names[:n_keep])
        else:
            raise ValueError(f"Unknown mode: {mode}")

        self.all_datasets = {rid: ds for rid, ds in datasets.items() if rid in chosen}
        assert self.all_datasets, "No datasets left after applying mode split."
        self.datasets = dict(self.all_datasets)
        self.runtime_excluded_episode_hashes = set()
        self.pending_excluded_episode_hashes = set()
        self.index_map = []
        self._rebuild_index_map()

        super().__init__()

        raw_embodiment = next(iter(self.datasets.values())).embodiment

        _EMBODIMENT_ALIASES = {
            "scale": "scale_bimanual",
            "aria": "aria_bimanual",
            "eva": "eva_bimanual",
            "mecka": "mecka_bimanual",
        }
        self.embodiment = _EMBODIMENT_ALIASES.get(
            raw_embodiment.lower(), raw_embodiment
        )

    def _rebuild_index_map(self):
        self.datasets = {
            rid: ds
            for rid, ds in self.all_datasets.items()
            if rid not in self.runtime_excluded_episode_hashes
        }
        assert self.datasets, "No datasets left after runtime exclusions."

        self.index_map = []
        for dataset_name, dataset in self.datasets.items():
            for local_idx in range(len(dataset)):
                self.index_map.append((dataset_name, local_idx))

    def queue_exclude_episode_hashes(self, episode_hashes: list[str] | set[str]):
        for episode_hash in episode_hashes:
            if (
                episode_hash in self.all_datasets
                and episode_hash not in self.runtime_excluded_episode_hashes
            ):
                self.pending_excluded_episode_hashes.add(episode_hash)

    def peek_pending_exclusions(self) -> list[str]:
        return sorted(self.pending_excluded_episode_hashes)

    def apply_pending_exclusions(self) -> list[str]:
        if not self.pending_excluded_episode_hashes:
            return []

        applied = sorted(
            self.pending_excluded_episode_hashes - self.runtime_excluded_episode_hashes
        )
        if not applied:
            self.pending_excluded_episode_hashes.clear()
            return []

        self.runtime_excluded_episode_hashes.update(applied)
        self.pending_excluded_episode_hashes.clear()
        self._rebuild_index_map()
        return applied

    def __len__(self) -> int:
        return len(self.index_map)

    def __getitem__(self, idx):
        dataset_name, local_idx = self.index_map[idx]
        data = self.datasets[dataset_name][local_idx]

        robot_name = self.embodiment  # already normalized by __init__

        from egomimic.rldb.embodiment.embodiment import get_embodiment_id

        data["metadata.robot_name"] = robot_name
        data["embodiment"] = robot_name
        data["metadata.embodiment"] = get_embodiment_id(robot_name.upper())
        data["metadata.episode_hash"] = dataset_name
        data["metadata.frame_idx"] = local_idx

        return data

    @classmethod
    def _from_resolver(cls, resolver: EpisodeResolver, **kwargs):
        """
        create a MultiDataset from an EpisodeResolver.

        Args:
            resolver (EpisodeResolver): The resolver instance to use for loading datasets.
            embodiment: The embodiment identifier to use for resolving datasets.
            **kwargs: Keyword args forwarded to resolver (e.g., filters,
                sync_from_s3) and MultiDataset constructor (e.g., mode, percent,
                key_map, valid_ratio).
        Returns:
            MultiDataset: The constructed multi-dataset.
        """
        # TODO add key_map and transform pass to children

        sync_from_s3 = kwargs.pop("sync_from_s3", False)
        filters = kwargs.pop("filters", {}) or {}

        if isinstance(resolver, LocalEpisodeResolver):
            resolved = resolver.resolve(
                sync_from_s3=sync_from_s3,
                filters=filters,
            )
        else:
            resolved = resolver.resolve(filters=filters)

        return cls(datasets=resolved, **kwargs)


class ZarrDataset(torch.utils.data.Dataset):
    """
    Base Zarr Dataset object, Just intializes as pass through to read from zarr episode
    """

    def __init__(
        self,
        Episode_path: str | Path,
        key_map: dict,
        transform_list: list | None = None,
    ):
        """
        Args:
            episode_path: just a path to the designated zarr episode
            key_map: dict mapping from dataset keys to zarr keys and horizon info, e.g. {"obs/image/front": {"zarr_key": "observations.images.front", "horizon": 4}, ...}
            transform_list: list of Transform objects to apply to the data after loading, e.g. for action chunk transformations. Should be in order of application.
        """
        self.episode_path = Episode_path
        self.metadata = None
        self._image_keys = None  # Lazy-loaded set of JPEG-encoded keys
        self._json_keys = None  # Lazy-loaded set of JSON-encoded keys
        self._annotations = None
        self.init_episode()

        self.key_map = key_map
        self.transform = transform_list
        super().__init__()

    def init_episode(self):
        """
        inits the zarr episode and all the metadata associated, as well as total_frames for len
        """
        self.episode_reader = ZarrEpisode(self.episode_path)
        self.metadata = self.episode_reader.metadata
        self.total_frames = self.metadata["total_frames"]
        self.embodiment = self.metadata["embodiment"]
        self.keys_dict = {k: (0, None) for k in self.episode_reader._collect_keys()}

        # Detect JPEG-encoded image keys from metadata
        self._image_keys = self._detect_image_keys()
        self._json_keys = self._detect_json_keys()

    def _detect_image_keys(self) -> set[str]:
        """
        Detect which keys contain JPEG-encoded image data from metadata.

        Returns:
            Set of keys containing JPEG data
        """
        features = self.metadata.get("features", {})
        return {key for key, info in features.items() if info.get("dtype") == "jpeg"}

    def _detect_json_keys(self) -> set[str]:
        """
        Detect keys containing JSON-encoded bytes from metadata.

        Returns:
            Set of keys containing JSON payloads.
        """
        features = self.metadata.get("features", {})
        return {key for key, info in features.items() if info.get("dtype") == "json"}

    @staticmethod
    def _decode_json_entry(value):
        if isinstance(value, np.void):
            value = value.item()
        if isinstance(value, memoryview):
            value = value.tobytes()
        if isinstance(value, bytearray):
            value = bytes(value)
        if isinstance(value, bytes):
            return json.loads(value.decode("utf-8"))
        if isinstance(value, str):
            return json.loads(value)
        return value

    def _load_annotations(self) -> list[dict]:
        """
        Load and cache decoded language annotations.

        Expected format per entry:
            {"text": str, "start_idx": int, "end_idx": int}
        """
        if self._annotations is not None:
            return self._annotations

        raw = self.episode_reader._store["annotations"][:]

        decoded = [self._decode_json_entry(x) for x in raw]
        self._annotations = [d for d in decoded if isinstance(d, dict)]
        return self._annotations

    def _annotation_text_for_frame(self, frame_idx: int) -> str:
        """
        Resolve language annotation text for a frame from span annotations.
        """
        annotations = self._load_annotations()
        for ann in annotations:
            start_idx = int(ann.get("start_idx", -1))
            end_idx = int(ann.get("end_idx", -1))
            if start_idx <= frame_idx <= end_idx:
                return str(ann.get("text", ""))
        return ""

    def __len__(self) -> int:
        return self.total_frames

    def _pad_sequences(self, data, horizon: int | None) -> dict:
        if horizon is None:
            return data

        # Note that k is zarr key
        for k in data:
            if isinstance(data[k], np.ndarray):
                seq_len = data[k].shape[0]
                if seq_len < horizon:
                    # Pad by repeating the last frame
                    pad_len = horizon - seq_len
                    last_frame = data[k][-1:]  # Keep dims: (1, action_dim)
                    padding = np.repeat(last_frame, pad_len, axis=0)
                    data[k] = np.concatenate([data[k], padding], axis=0)

        return data

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        # Build keys_dict with ranges based on whether action chunking is enabled
        return self.get_item_keys(idx, keys=None)

    def get_item_keys(self, idx: int, keys) -> dict[str, torch.Tensor]:
        requested = self._normalize_keys_arg(keys)
        out = {}

        # Load ALL keys first
        for k in requested:
            if k not in self.key_map:
                raise KeyError(
                    f"Unknown key '{k}'. Available: {list(self.key_map.keys())}"
                )

            zarr_key = self.key_map[k]["zarr_key"]
            horizon = self.key_map[k].get("horizon", None)

            if horizon is not None:
                end_idx = min(idx + horizon, self.total_frames)
                interval = (idx, end_idx)
            else:
                interval = (idx, None)

            raw = self.episode_reader.read({zarr_key: interval})
            self._pad_sequences(raw, horizon)
            val = raw[zarr_key]

            if zarr_key in self._image_keys:
                if (
                    isinstance(val, np.ndarray)
                    and val.dtype == object
                    and val.ndim == 1
                ):
                    decoded_seq = []
                    for jpeg_bytes in val:
                        img = simplejpeg.decode_jpeg(jpeg_bytes, colorspace="RGB")
                        decoded_seq.append(np.transpose(img, (2, 0, 1)) / 255.0)
                    val = np.stack(decoded_seq, axis=0)
                else:
                    img = simplejpeg.decode_jpeg(val, colorspace="RGB")
                    val = np.transpose(img, (2, 0, 1)) / 255.0
                import torchvision.transforms.functional as TF

                val_t = torch.from_numpy(val.copy()).float()
                if val_t.ndim == 3:
                    val_t = TF.resize(val_t, [224, 224], antialias=True)
                elif val_t.ndim == 4:
                    val_t = torch.stack(
                        [TF.resize(f, [224, 224], antialias=True) for f in val_t]
                    )
                out[k] = val_t
            else:
                out[k] = val

        # Apply transforms ONCE, after all keys are loaded
        if self.transform:
            for transform in self.transform or []:
                try:
                    out = transform.transform(out)
                except Exception as e:
                    logger.error(f"Error transforming data: {e}")
                    logger.error(f"Data keys: {list(out.keys())}")
                    logger.error(f"Transform: {transform}")
                    logger.error(f"Error: {e}")
                    if idx == 0:
                        logger.error("Error in first frame")
                        raise e
                    else:
                        return self.get_item_keys(0, keys)

        for k, v in out.items():
            if isinstance(v, np.ndarray):
                out[k] = torch.from_numpy(v).to(torch.float32)

        return out

    def _normalize_keys_arg(self, keys):
        """
        Normalize keys argument:
          None -> all dataset keys
          str  -> single key
          Iterable[str] -> list of keys
        """
        if keys is None:
            return list(self.key_map.keys())
        if isinstance(keys, str):
            return [keys]
        if isinstance(keys, Iterable):
            return list(keys)
        raise TypeError(f"keys must be None, str, or iterable[str], got {type(keys)}")


class ZarrEpisode:
    """
    Lightweight wrapper around a single Zarr episode store.
    Designed for efficient PyTorch DataLoader usage with direct store access.
    """

    __slots__ = (
        "_path",
        "_store",
        "_store_opened",
        "_store_pid",
        "metadata",
        "keys",
    )

    def __init__(self, path: str | Path):
        """
        Initialize ZarrEpisode wrapper.
        Args:
            path: Path to the .zarr episode directory
        """
        self._path = str(path) if self._is_remote_path(path) else str(Path(path))
        self._store = None
        self._store_opened = False
        self._store_pid = None
        self.metadata = {}
        self._ensure_store()
        self.metadata = dict(self._store.attrs)
        self.keys = self.metadata["features"]

    @staticmethod
    def _is_remote_path(path: str | Path) -> bool:
        parsed = urlparse(str(path))
        return bool(parsed.scheme and parsed.scheme != "file")

    @staticmethod
    def _build_remote_store(path: str):
        load_env()
        endpoint_url = os.environ.get("R2_ENDPOINT_URL")
        access_key_id = os.environ.get("R2_ACCESS_KEY_ID")
        secret_access_key = os.environ.get("R2_SECRET_ACCESS_KEY")
        if not all([endpoint_url, access_key_id, secret_access_key]):
            raise ValueError(
                "R2 credentials missing. Ensure ~/.egoverse_env has "
                "R2_ENDPOINT_URL, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY."
            )

        try:
            import s3fs
        except ImportError as e:
            raise ImportError(
                "s3fs is required for direct R2/S3 Zarr streaming."
            ) from e

        parsed = urlparse(path)
        bucket = parsed.netloc
        key = parsed.path.lstrip("/").rstrip("/")
        fs = s3fs.S3FileSystem(
            key=access_key_id,
            secret=secret_access_key,
            client_kwargs={
                "endpoint_url": endpoint_url,
                "region_name": "auto",
            },
        )
        mapper = fs.get_mapper(f"{bucket}/{key}")
        fsspec_store_cls = getattr(zarr.storage, "FsspecStore", None)
        if fsspec_store_cls is None:
            return mapper
        if hasattr(fsspec_store_cls, "from_mapper"):
            return fsspec_store_cls.from_mapper(mapper)
        return fsspec_store_cls(mapper)

    def _ensure_store(self):
        current_pid = os.getpid()
        if self._store_opened and self._store_pid == current_pid:
            return

        if self._is_remote_path(self._path):
            store = self._build_remote_store(self._path)
            self._store = zarr.open_group(store=store, mode="r")
        else:
            self._store = zarr.open_group(self._path, mode="r")
        self._store_opened = True
        self._store_pid = current_pid

    def __getstate__(self):
        return {
            "_path": self._path,
            "_store": None,
            "_store_opened": False,
            "_store_pid": None,
            "metadata": self.metadata,
            "keys": self.keys,
        }

    def __setstate__(self, state):
        self._path = state["_path"]
        self._store = None
        self._store_opened = False
        self._store_pid = None
        self.metadata = state["metadata"]
        self.keys = state["keys"]

    def read(
        self, keys_with_ranges: dict[str, tuple[int, int | None]]
    ) -> dict[str, np.ndarray]:
        """
        Read data for specified keys, each with their own index or range.
        Args:
            keys_with_ranges: Dictionary mapping keys to (start, end) tuples.
                - start: Starting frame index
                - end: Ending frame index (exclusive). If None, reads single frame at start.
        Returns:
            Dictionary mapping keys to numpy arrays
        Example:
            >>> episode.read({
            ...     "obs/image": (0, 10),      # Read frames 0-10
            ...     "actions": (5, 15),        # Read frames 5-15
            ...     "rewards": (20, None),     # Read single frame at index 20
            ... })
        """
        self._ensure_store()
        result = {}
        for key, (start, end) in keys_with_ranges.items():
            arr = self._store[key]
            if end is not None:
                data = arr[start:end]
            else:
                # Single frame read - use slicing to avoid 0D array issues with VariableLengthBytes
                # arr[start:start+1] gives us a 1D array, then [0] extracts the actual object
                data = arr[start : start + 1][0]
            result[key] = data
        return result

    def _collect_keys(self) -> list[str]:
        """
        Collect all array keys from the store.
        Returns:
            List of array keys (flat structure with dot-separated names)
        """
        if isinstance(self.keys, dict):
            return list(self.keys.keys())
        return list(self.keys)

    def __len__(self) -> int:
        """
        Get total number of frames in the episode.
        Returns:
            Number of frames
        """
        return self.metadata["total_frames"]

    def __repr__(self) -> str:
        """String representation of the episode."""
        return f"ZarrEpisode(path={self._path}, frames={len(self)})"
