#!/usr/bin/env python3
import argparse
import json
import math
import shutil
from pathlib import Path

REQUIRED_KEYS = (
    "images.front_1",
    "right.obs_ee_pose",
    "left.obs_ee_pose",
    "obs_head_pose",
)


def _is_bad_error_value(value) -> bool:
    if value is None:
        return False
    if isinstance(value, float) and math.isnan(value):
        return False
    if isinstance(value, str):
        return value.strip().lower() not in {"", "none", "nan", "null"}
    return bool(value)


def _expected_chunk_paths(chunks_dir: Path, shape: list[int], chunk_shape: list[int]):
    if not shape or not chunk_shape:
        return []
    chunk_counts = [
        int(math.ceil(dim / chunk_dim)) for dim, chunk_dim in zip(shape, chunk_shape)
    ]
    if len(chunk_counts) == 1:
        return [chunks_dir / str(i) for i in range(chunk_counts[0])]
    if len(chunk_counts) == 2:
        return [
            chunks_dir / str(i) / str(j)
            for i in range(chunk_counts[0])
            for j in range(chunk_counts[1])
        ]
    return []


def _edge_chunk_paths(chunks_dir: Path, shape: list[int], chunk_shape: list[int]):
    if not shape or not chunk_shape:
        return []
    chunk_counts = [
        int(math.ceil(dim / chunk_dim)) for dim, chunk_dim in zip(shape, chunk_shape)
    ]
    if len(chunk_counts) == 1:
        return [chunks_dir / "0", chunks_dir / str(chunk_counts[0] - 1)]
    if len(chunk_counts) == 2:
        return [
            chunks_dir / "0" / "0",
            chunks_dir / str(chunk_counts[0] - 1) / str(chunk_counts[1] - 1),
        ]
    return []


def check_episode(
    path: Path,
    read_edges: bool = True,
    check_chunks: bool = False,
    min_frames: int = 100,
) -> tuple[bool, str]:
    try:
        with (path / "zarr.json").open() as f:
            root_meta = json.load(f)
    except Exception as exc:
        return False, f"cannot read root zarr.json: {type(exc).__name__}: {exc}"

    try:
        attrs = root_meta.get("attributes", {})
        total_frames = int(attrs.get("total_frames", 0))
    except Exception:
        total_frames = 0
    if total_frames <= 0:
        return False, "missing or invalid attributes.total_frames"
    if total_frames < min_frames:
        return False, f"too few frames: {total_frames} < {min_frames}"

    if attrs.get("robot_name") not in {None, "scale_bimanual"}:
        return False, f"unexpected robot_name: {attrs.get('robot_name')!r}"
    if attrs.get("is_deleted") is True:
        return False, "episode is marked deleted"
    for error_key in ("processing_error", "zarr_processing_error"):
        if _is_bad_error_value(attrs.get(error_key)):
            return False, f"{error_key} is set: {attrs.get(error_key)!r}"

    features = attrs.get("features", {})
    missing = [key for key in REQUIRED_KEYS if key not in features]
    if missing:
        return False, f"missing required features: {missing}"

    for key in REQUIRED_KEYS:
        arr_dir = path / key
        arr_meta_path = arr_dir / "zarr.json"
        chunks_dir = arr_dir / "c"
        if not arr_meta_path.is_file():
            return False, f"{key} missing zarr.json"
        if not chunks_dir.is_dir():
            return False, f"{key} missing chunk directory c/"
        try:
            with arr_meta_path.open() as f:
                arr_meta = json.load(f)
            shape = arr_meta.get("shape", [])
            chunk_shape = (
                arr_meta.get("chunk_grid", {})
                .get("configuration", {})
                .get("chunk_shape", [])
            )
            if not shape or int(shape[0]) < total_frames:
                return (
                    False,
                    f"{key} shape {shape} is shorter than total_frames {total_frames}",
                )
            if read_edges:
                edge_chunks = _edge_chunk_paths(chunks_dir, shape, chunk_shape)
                if not edge_chunks:
                    return False, f"{key} cannot determine edge chunks"
                missing_edges = [p for p in edge_chunks if not p.is_file()]
                if missing_edges:
                    rel = missing_edges[0].relative_to(path)
                    return False, f"{key} missing edge chunk: {rel}"
                empty_edges = [p for p in edge_chunks if p.stat().st_size <= 0]
                if empty_edges:
                    rel = empty_edges[0].relative_to(path)
                    return False, f"{key} edge chunk is empty: {rel}"
            if check_chunks:
                expected = _expected_chunk_paths(chunks_dir, shape, chunk_shape)
                if not expected:
                    return False, f"{key} cannot determine expected chunks"
                missing_chunks = [p for p in expected if not p.is_file()]
                if missing_chunks:
                    rel = missing_chunks[0].relative_to(path)
                    return (
                        False,
                        f"{key} missing {len(missing_chunks)} chunks; first: {rel}",
                    )
                empty_chunks = [p for p in expected if p.stat().st_size <= 0]
                if empty_chunks:
                    rel = empty_chunks[0].relative_to(path)
                    return False, f"{key} contains empty chunks; first: {rel}"
        except Exception as exc:
            return False, f"{key} metadata check failed: {type(exc).__name__}: {exc}"

    return True, f"ok ({total_frames} frames)"


def main():
    parser = argparse.ArgumentParser(description="Validate cached Scale Zarr episodes.")
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("/data/sybeuret/scale_zarr_cache"),
    )
    parser.add_argument(
        "--quarantine-dir",
        type=Path,
        default=None,
        help="Move bad episode directories here. If omitted, only report.",
    )
    parser.add_argument(
        "--no-read-edges",
        action="store_true",
        help="Only check metadata and array presence, without reading first/last samples.",
    )
    parser.add_argument(
        "--check-chunks",
        action="store_true",
        help="Check that every expected chunk file exists for required arrays.",
    )
    parser.add_argument(
        "--min-frames",
        type=int,
        default=100,
        help="Minimum required episode length.",
    )
    args = parser.parse_args()

    episode_dirs = sorted(p for p in args.cache_dir.iterdir() if p.is_dir())
    print(f"Checking {len(episode_dirs)} episode directories in {args.cache_dir}")

    bad: list[tuple[Path, str]] = []
    for idx, path in enumerate(episode_dirs, start=1):
        ok, reason = check_episode(
            path,
            read_edges=not args.no_read_edges,
            check_chunks=args.check_chunks,
            min_frames=args.min_frames,
        )
        if not ok:
            bad.append((path, reason))
            print(f"[BAD {idx:04d}] {path.name}: {reason}")
        elif idx <= 5 or idx % 50 == 0:
            print(f"[OK  {idx:04d}] {path.name}: {reason}")

    print()
    print(f"Valid episodes: {len(episode_dirs) - len(bad)}")
    print(f"Bad episodes:   {len(bad)}")

    if bad and args.quarantine_dir is not None:
        args.quarantine_dir.mkdir(parents=True, exist_ok=True)
        for path, reason in bad:
            dst = args.quarantine_dir / path.name
            print(f"Moving bad episode {path.name} -> {dst} ({reason})")
            if dst.exists():
                raise FileExistsError(f"Quarantine destination already exists: {dst}")
            shutil.move(str(path), str(dst))

    if bad:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
