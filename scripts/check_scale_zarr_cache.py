#!/usr/bin/env python3
import argparse
import json
import shutil
from pathlib import Path

REQUIRED_KEYS = (
    "images.front_1",
    "right.obs_ee_pose",
    "left.obs_ee_pose",
    "obs_head_pose",
)


def check_episode(path: Path, read_edges: bool = True) -> tuple[bool, str]:
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
            if not shape or int(shape[0]) < total_frames:
                return (
                    False,
                    f"{key} shape {shape} is shorter than total_frames {total_frames}",
                )
            if read_edges and not any(chunks_dir.iterdir()):
                return False, f"{key} chunk directory is empty"
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
    args = parser.parse_args()

    episode_dirs = sorted(p for p in args.cache_dir.iterdir() if p.is_dir())
    print(f"Checking {len(episode_dirs)} episode directories in {args.cache_dir}")

    bad: list[tuple[Path, str]] = []
    for idx, path in enumerate(episode_dirs, start=1):
        ok, reason = check_episode(path, read_edges=not args.no_read_edges)
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
