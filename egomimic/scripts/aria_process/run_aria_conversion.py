#!/usr/bin/env python3
"""
run_aria_conversion.py
─────────────────────────────────────────────────────────────────────────
 • Scans every task folder listed in /mnt/raw/task_map.csv
 • Spawns one Ray task *per VRS bundle* that hasn’t been converted yet
 • Each task symlinks the three required inputs ( .vrs , .vrs.json , mps_* )
   into ~/temp_mps_processing/<random>  ➜  runs ``lerobot_job`` there
 • Results are written to /mnt/processed/<task>/<stem>_processed/…
 • A **single global progress file** is maintained:
       /mnt/processed/vrs_conversion_status.csv
   The file is opened in *append* mode under a local lock so it works on S3-FUSE.
"""

import argparse, csv, json, os, shutil, sys, traceback, uuid
from pathlib import Path
from filelock import FileLock
import ray

# ───────────── paths ──────────────────────────────────────────────
RAW_ROOT        = Path("/mnt/raw")
PROCESSED_ROOT  = Path("/mnt/processed")

TASK_MAP_CSV    = RAW_ROOT / "task_map.csv"                 # cols: task,arm
GLOBAL_STATUS   = PROCESSED_ROOT / "vrs_conversion_status.csv"

TMP_ROOT        = Path.home() / "temp_mps_processing"       # per-job workspace
LOCK_PATH       = Path("/tmp/vrs_status.lock")              # local filesystem
lock            = FileLock(str(LOCK_PATH))

# helper that actually calls aria_to_lerobot.py
from aria_helper import lerobot_job

# ───────────── helpers ────────────────────────────────────────────
def load_task_map() -> dict[str, str]:
    """Read task_map.csv → {task: arm} (arm lowered)."""
    with TASK_MAP_CSV.open() as f:
        return {r["task"].strip(): r["arm"].strip().lower()
                for r in csv.DictReader(f)}

def already_done() -> set[str]:
    """Return set of ‘task/stem’ keys already processed (from global CSV)."""
    if not GLOBAL_STATUS.exists():
        return set()
    with GLOBAL_STATUS.open() as f:
        return {f"{r['task']}/{r['vrs']}" for r in csv.DictReader(f)}

def vrs_bundles(task_dir: Path):
    """Yield (vrs_file, json_file, mps_dir) triples that pass integrity checks."""
    for vrs in task_dir.glob("*.vrs"):
        stem  = vrs.stem
        jsonf = task_dir / f"{stem}.vrs.json"
        mps   = task_dir / f"mps_{stem}_vrs"
        if not (jsonf.exists() and mps.is_dir()
                and (mps / "hand_tracking/wrist_and_palm_poses.csv").exists()
                and (mps / "slam/closed_loop_trajectory.csv").exists()):
            continue
        yield vrs, jsonf, mps

def append_status(row: dict):
    """Append one row to the global CSV in a mount-s3-safe way."""
    with lock:
        new_file = not GLOBAL_STATUS.exists()
        with GLOBAL_STATUS.open("a", newline="") as f:
            w = csv.DictWriter(
                f, fieldnames=["task", "vrs", "total_frames", "output_path"])
            if new_file:
                w.writeheader()
            w.writerow(row)

# ───────────── Ray remote task ────────────────────────────────────
@ray.remote(num_cpus=8, memory=16 * 1024**3)
def convert_one(tmp_dir: str, out_dir: str,
                dataset_name: str, arm: str) -> tuple[str, int]:
    """
    Run conversion; return (dataset_path, total_frames).
    On failure → frames = -1.
    """
    out_dir  = Path(out_dir)
    ds_path  = out_dir / dataset_name
    try:
        print(f"[INFO] Converting → {ds_path}", flush=True)

        lerobot_job(raw_path=tmp_dir,
                    output_dir=str(out_dir),
                    dataset_name=dataset_name,
                    arm=arm,
                    description="")

        info_p = ds_path / "meta/info.json"
        frames = -1
        if info_p.exists():
            frames = int(json.loads(info_p.read_text()).get("total_frames", -1))

        print(f"[INFO] Done   → {ds_path} ({frames} frames)", flush=True)
        return str(ds_path), frames

    except Exception as exc:
        traceback.print_exc(file=sys.stdout)          # <── full stack once
        print(f"[ERROR] {ds_path} failed: {exc}", flush=True)
        return str(ds_path), -1

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

# ───────────── main driver ────────────────────────────────────────
def launch(dry_run: bool = False):
    already = already_done()
    pending: dict[ray.ObjectRef, tuple[str, str, str]] = {}  # ref → (task, vrs, ds_path)

    for task, arm in load_task_map().items():
        task_dir = RAW_ROOT / task
        for vrs, jsonf, mps in vrs_bundles(task_dir):
            key = f"{task}/{vrs.stem}"
            if key in already:
                continue

            # make a temp dir of symlinks
            tmp = TMP_ROOT / f"{vrs.stem}-{uuid.uuid4().hex[:6]}"
            tmp.mkdir(parents=True, exist_ok=True)
            for src in (vrs, jsonf, mps):
                os.symlink(src, tmp / src.name, target_is_directory=src.is_dir())

            out_dir  = PROCESSED_ROOT / task
            dataset  = f"{vrs.stem}_processed"
            ds_path  = out_dir / dataset

            if dry_run:
                print(f"[DRY-RUN] {task} → {ds_path}  (arm={arm})")
                shutil.rmtree(tmp, ignore_errors=True)
            else:
                ref = convert_one.remote(str(tmp), str(out_dir), dataset, arm)
                pending[ref] = (task, vrs.stem, str(ds_path))

    if dry_run or not pending:
        print("Dry run complete." if dry_run else "Nothing to do.")
        return

    print(f"Submitted {len(pending)} jobs to Ray…")

    while pending:
        finished, _ = ray.wait(list(pending.keys()), num_returns=1)
        ref = finished[0]
        ds_path, frames = ray.get(ref)
        task, vrs_stem, _ = pending.pop(ref)

        append_status(
            dict(task=task,
                 vrs=vrs_stem,
                 total_frames=frames,
                 output_path=ds_path))
        print(f"[LOG] {task}/{vrs_stem} → {frames} frames")

# ───────────── CLI entry ──────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="List conversions without actually running them")
    args = parser.parse_args()

    ray.init(address="auto")
    launch(dry_run=args.dry_run)
