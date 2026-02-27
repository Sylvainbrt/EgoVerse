#!/usr/bin/env python3
"""
Scale SFS -> EgoVerse Zarr converter.

Output keys per episode:
    left.obs_ee_pose                 (T, 7)         xyz + quat(w, x, y, z)
    right.obs_ee_pose                (T, 7)         xyz + quat(w, x, y, z)
    left.obs_keypoints               (T, 63)        21 keypoints * 3 (xyz)
    right.obs_keypoints              (T, 63)        21 keypoints * 3 (xyz)
    left.obs_wrist_pose              (T, 7)         xyz + quat(w, x, y, z)
    right.obs_wrist_pose             (T, 7)         xyz + quat(w, x, y, z)
    obs_head_pose                    (T, 7)         xyz + quat(w, x, y, z)
    images.front_1                   (T, H, W, 3)   JPEG-compressed by ZarrWriter

Filtering:
  Frames are dropped when:
    - Either hand has missing keypoint predictions
    - Frame falls within a Hand Tracking Error annotation range
    - Frame falls within an Inactive Time collector issue range
  Only contiguous runs of valid frames are kept as sub-episodes.

Usage:
  python sfs_to_egoverse_zarr.py --task-ids TASK1 TASK2 --output-dir ./zarr_out
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import simplejpeg
from decord import VideoReader, cpu as decord_cpu
from scipy.spatial.transform import Rotation as R

from egomimic.rldb.zarr.zarr_writer import ZarrWriter

from scale_api import (
    download_from_simple_response_dict,
    get_intrinsics,
    get_simple_response_dict_egocentric,
    load_scene,
)
from sfs_data import (
    INVALID_VALUE,
    FrameData,
    SFSDataExtractor,
    batch_pose6_to_pose7,
    compute_palm_6dof,
    compute_wrist_6dof,
)

SUB_EPISODE_LENGTH = 300
MIN_EPISODE_FRAMES = 10
IMAGE_SIZE = (640, 480)  # (W, H) for cv2.resize


# ---------------------------------------------------------------------------
# Video / image helpers
# ---------------------------------------------------------------------------


def _get_video_frame_count(video_path: str) -> int:
    """Get frame count without decoding the video."""
    vr = VideoReader(video_path, ctx=decord_cpu())
    return len(vr)


def _decode_selected_frames(
    video_path: str,
    indices: list[int],
    chunk_size: int = 500,
    resize: tuple[int, int] | None = IMAGE_SIZE,
) -> dict[int, np.ndarray]:
    """Batch-decode only the requested frame indices via decord.

    Decodes in chunks and eagerly resizes to *resize* (W, H) to keep
    memory usage bounded.  Returns a dict mapping frame index to RGB uint8.
    """
    if not indices:
        return {}
    indices_sorted = sorted(set(indices))
    vr = VideoReader(video_path, ctx=decord_cpu())
    max_idx = len(vr) - 1
    valid = [i for i in indices_sorted if i <= max_idx]
    if not valid:
        return {}
    result: dict[int, np.ndarray] = {}
    for start in range(0, len(valid), chunk_size):
        chunk_indices = valid[start : start + chunk_size]
        batch = vr.get_batch(chunk_indices).asnumpy()
        for i, t in enumerate(chunk_indices):
            frame = batch[i]
            if resize is not None:
                frame = cv2.resize(frame, resize, interpolation=cv2.INTER_LINEAR)
            result[t] = frame
        del batch
    return result


def _resize_and_encode(frame: np.ndarray) -> tuple[tuple[int, ...], bytes]:
    """Resize frame to IMAGE_SIZE and JPEG-encode it. GIL-releasing."""
    resized = cv2.resize(frame, IMAGE_SIZE, interpolation=cv2.INTER_LINEAR)
    jpeg = simplejpeg.encode_jpeg(
        resized, quality=ZarrWriter.JPEG_QUALITY, colorspace="RGB"
    )
    return resized.shape, jpeg


# ---------------------------------------------------------------------------
# Preview MP4
# ---------------------------------------------------------------------------


def _save_preview_mp4(
    image_frames: list[np.ndarray],
    output_path: Path,
    fps: int = 30,
) -> None:
    """Save a half-resolution H.264 preview video via ffmpeg."""
    if not image_frames:
        return
    H, W = image_frames[0].shape[:2]
    out_w, out_h = (W // 2) & ~1, (H // 2) & ~1
    if out_w <= 0 or out_h <= 0:
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        print(f"  [preview] ffmpeg not found, skipping {output_path.name}")
        return

    cmd = [
        ffmpeg, "-y",
        "-f", "rawvideo", "-vcodec", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{out_w}x{out_h}", "-r", str(fps),
        "-i", "-",
        "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-profile:v", "baseline", "-level", "3.0",
        "-movflags", "+faststart", "-preset", "veryfast", "-crf", "23",
        str(output_path),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    for frame in image_frames:
        resized = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)
        proc.stdin.write(resized.tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        stderr = proc.stderr.read().decode(errors="replace")
        print(f"  [preview] ffmpeg failed for {output_path.name}: {stderr[:200]}")
    else:
        print(f"  [preview] saved {output_path.name}")


# ---------------------------------------------------------------------------
# Frame validity
# ---------------------------------------------------------------------------


def _build_validity_mask(frames: list[FrameData], video_frame_count: int) -> np.ndarray:
    """Boolean mask: True = frame has valid hand tracking and no errors."""
    n = len(frames)
    mask = np.ones(n, dtype=bool)
    drop_reasons: dict[str, int] = {
        "missing_left_hand": 0,
        "missing_right_hand": 0,
        "hand_tracking_error": 0,
        "inactive_time": 0,
        "beyond_video": 0,
    }
    for i, frame in enumerate(frames):
        if frame.hand_keypoints.left is None:
            mask[i] = False
            drop_reasons["missing_left_hand"] += 1
            continue
        if frame.hand_keypoints.right is None:
            mask[i] = False
            drop_reasons["missing_right_hand"] += 1
            continue
        if frame.hand_tracking_error is not None:
            mask[i] = False
            drop_reasons["hand_tracking_error"] += 1
            continue
        if (
            frame.collector_issue is not None
            and frame.collector_issue.get("issue_type") == "Inactive Time"
        ):
            mask[i] = False
            drop_reasons["inactive_time"] += 1
            continue
        if i >= video_frame_count:
            mask[i] = False
            drop_reasons["beyond_video"] += 1

    valid_count = int(mask.sum())
    total = len(frames)
    dropped = total - valid_count
    print(f"  Validity: {valid_count}/{total} frames kept ({dropped} dropped)")
    for reason, count in drop_reasons.items():
        if count > 0:
            print(f"    {reason}: {count}")
    return mask


def _contiguous_runs(mask: np.ndarray, min_length: int) -> list[list[int]]:
    """Extract contiguous runs of True indices from a boolean mask."""
    runs: list[list[int]] = []
    current: list[int] = []
    for i, val in enumerate(mask):
        if val:
            current.append(i)
        else:
            if len(current) >= min_length:
                runs.append(current)
            current = []
    if len(current) >= min_length:
        runs.append(current)
    return runs


# ---------------------------------------------------------------------------
# Language annotations
# ---------------------------------------------------------------------------


def _build_language_annotations(sub_frames: list[FrameData]) -> list[tuple[str, int, int]]:
    rows: list[tuple[str, int, int]] = []
    current_text: str | None = None
    start_idx: int | None = None
    for idx, frame in enumerate(sub_frames):
        label = frame.subgoal.get("text", "").strip() if frame.subgoal else ""
        text = label if label else None
        if text != current_text:
            if current_text is not None and start_idx is not None:
                rows.append((current_text, start_idx, idx - 1))
            current_text = text
            start_idx = idx if text is not None else None
    if current_text is not None and start_idx is not None:
        rows.append((current_text, start_idx, len(sub_frames) - 1))
    return rows


def _task_description(frames: list[FrameData], demo_meta: dict[str, Any]) -> str:
    candidate = str(demo_meta.get("Demonstration", "")).strip()
    if candidate:
        return candidate
    skip = {"Inactive Time", "Collector Issue", "inactive time", "collector issue"}
    for frame in frames:
        for ann in frame.text_annotations:
            text = str(ann.get("text", "")).strip()
            label = str(ann.get("label", "")).strip()
            if text and text not in skip and label not in skip:
                return text
    return "Unknown task"


# ---------------------------------------------------------------------------
# Core conversion
# ---------------------------------------------------------------------------


def convert_task_to_zarr(
    task_id: str,
    output_dir: str,
    download_dir: str,
    robot_type: str = "scale_bimanual",
    fps: int = 30,
    img_workers: int | None = None,
) -> dict[str, Any]:
    """Convert one Scale task to one or more Zarr episodes.

    Returns a dict with keys: episodes, folder, task_desc, total_frames, output_dir.
    """
    t_start = time.perf_counter()
    if img_workers is None:
        img_workers = min(os.cpu_count() or 4, 8)

    print(f"[{task_id}] Fetching task metadata...")
    task_download_path = os.path.join(download_dir, task_id)
    os.makedirs(task_download_path, exist_ok=True)

    response = get_simple_response_dict_egocentric(task_id)
    if response is None:
        raise ValueError(f"Task {task_id} not found or Scale API failed")

    print(f"[{task_id}] Downloading files...")
    local_paths = download_from_simple_response_dict(task_download_path, response)
    sfs_path = local_paths.get("sfs")
    annotations_path = local_paths.get("annotations")
    video_path = local_paths.get("left_rectified") or local_paths.get("left_rgb")
    if not all([sfs_path, annotations_path, video_path]):
        raise ValueError(f"Missing SFS/annotation/video files for task {task_id}")

    def _nonempty(p: str | None) -> bool:
        return bool(p) and os.path.exists(p) and os.path.getsize(p) > 0

    if not (_nonempty(sfs_path) and _nonempty(annotations_path)):
        raise ValueError(f"Downloaded SFS/annotation files are empty for task {task_id}")

    print(f"[{task_id}] Loading SFS metadata...")
    try:
        extractor = SFSDataExtractor(sfs_path, annotations_path, video_path)
    except ValueError:
        print(f"[{task_id}] Load failed — re-downloading SFS + annotations...")
        for p in (sfs_path, annotations_path):
            if p and os.path.exists(p):
                os.remove(p)
        local_paths = download_from_simple_response_dict(task_download_path, response)
        sfs_path = local_paths.get("sfs")
        annotations_path = local_paths.get("annotations")
        video_path = local_paths.get("left_rectified") or local_paths.get("left_rgb")
        extractor = SFSDataExtractor(sfs_path, annotations_path, video_path)
    frames = extractor.extract_all_frames_metadata()
    n_frames = len(frames)
    if n_frames < MIN_EPISODE_FRAMES:
        raise ValueError(f"Task {task_id} has too few frames ({n_frames})")

    task_desc = _task_description(frames, extractor.demonstration_metadata)

    # Extract and scale intrinsics to output image resolution
    raw_intrinsics = get_intrinsics(
        load_scene(sfs_path), "left_rectified"
    )
    camera_intrinsics: dict[str, Any] | None = None
    if raw_intrinsics:
        orig_w = raw_intrinsics.get("width", 1920)
        orig_h = raw_intrinsics.get("height", 1200)
        sx = IMAGE_SIZE[0] / orig_w
        sy = IMAGE_SIZE[1] / orig_h
        camera_intrinsics = {
            "fx": raw_intrinsics["fx"] * sx,
            "fy": raw_intrinsics["fy"] * sy,
            "cx": raw_intrinsics["cx"] * sx,
            "cy": raw_intrinsics["cy"] * sy,
            "width": IMAGE_SIZE[0],
            "height": IMAGE_SIZE[1],
        }

    # ------------------------------------------------------------------
    # Probe video frame count
    # ------------------------------------------------------------------
    video_frame_count = _get_video_frame_count(video_path)
    print(f"[{task_id}] Video: {video_frame_count} frames  SFS: {n_frames} frames")
    if video_frame_count != n_frames:
        print(f"[{task_id}] WARNING: video/SFS frame count mismatch")

    # ------------------------------------------------------------------
    # Build per-frame validity mask and find contiguous runs
    # ------------------------------------------------------------------
    validity = _build_validity_mask(frames, video_frame_count)
    runs = _contiguous_runs(validity, min_length=MIN_EPISODE_FRAMES)
    if not runs:
        raise ValueError(f"Task {task_id} has no valid contiguous runs after filtering")

    print(f"[{task_id}] {len(runs)} contiguous run(s), "
          f"total valid frames: {sum(len(r) for r in runs)}")

    # ------------------------------------------------------------------
    # Precompute all per-frame data into dense arrays (no video needed)
    # ------------------------------------------------------------------
    left_world_6 = np.full((n_frames, 6), INVALID_VALUE, dtype=np.float32)
    right_world_6 = np.full((n_frames, 6), INVALID_VALUE, dtype=np.float32)
    left_wrist_6 = np.full((n_frames, 6), INVALID_VALUE, dtype=np.float32)
    right_wrist_6 = np.full((n_frames, 6), INVALID_VALUE, dtype=np.float32)
    left_kps = np.full((n_frames, 63), INVALID_VALUE, dtype=np.float32)
    right_kps = np.full((n_frames, 63), INVALID_VALUE, dtype=np.float32)
    head_pose_6 = np.zeros((n_frames, 6), dtype=np.float32)

    for i, frame in enumerate(frames):
        if frame.hand_keypoints.left is not None:
            left_world_6[i] = compute_palm_6dof(frame.hand_keypoints.left)
            left_wrist_6[i] = compute_wrist_6dof(frame.hand_keypoints.left)
            left_kps[i] = frame.hand_keypoints.left.flatten().astype(np.float32)
        if frame.hand_keypoints.right is not None:
            right_world_6[i] = compute_palm_6dof(frame.hand_keypoints.right, flip_x=True)
            right_wrist_6[i] = compute_wrist_6dof(frame.hand_keypoints.right, flip_x=True)
            right_kps[i] = frame.hand_keypoints.right.flatten().astype(np.float32)
        head_pose_6[i, :3] = frame.camera_pose.position.astype(np.float32)
        head_pose_6[i, 3:] = R.from_matrix(frame.camera_pose.rotation_matrix).as_euler(
            "ZYX", degrees=False
        ).astype(np.float32)

    left_world = batch_pose6_to_pose7(left_world_6)
    right_world = batch_pose6_to_pose7(right_world_6)
    left_wrist = batch_pose6_to_pose7(left_wrist_6)
    right_wrist = batch_pose6_to_pose7(right_wrist_6)
    head_pose_world = batch_pose6_to_pose7(head_pose_6)

    # ------------------------------------------------------------------
    # Split contiguous runs into sub-episodes and write
    # ------------------------------------------------------------------
    sub_episode_plans: list[list[int]] = []
    for run in runs:
        for ep_start in range(0, len(run), SUB_EPISODE_LENGTH):
            sub = run[ep_start : ep_start + SUB_EPISODE_LENGTH]
            if len(sub) >= MIN_EPISODE_FRAMES:
                sub_episode_plans.append(sub)

    folder = datetime.now(timezone.utc).strftime("%Y-%m-%d-%H-%M-%S-%f")
    task_output_dir = Path(output_dir) / folder
    task_output_dir.mkdir(parents=True, exist_ok=True)

    written = 0

    for sub in sub_episode_plans:
        decoded = _decode_selected_frames(video_path, sub)
        kept = [t for t in sub if t in decoded]
        if len(kept) < MIN_EPISODE_FRAMES:
            del decoded
            continue

        T = len(kept)
        kept_arr = np.array(kept)

        # All kept frames should have valid hand tracking;
        # replace any remaining per-keypoint INVALID_VALUE sentinels with 0.0
        left_curr_7 = np.where(
            left_world[kept_arr] >= INVALID_VALUE - 1, 0.0, left_world[kept_arr]
        ).astype(np.float32)
        right_curr_7 = np.where(
            right_world[kept_arr] >= INVALID_VALUE - 1, 0.0, right_world[kept_arr]
        ).astype(np.float32)
        left_wrist_curr_7 = np.where(
            left_wrist[kept_arr] >= INVALID_VALUE - 1, 0.0, left_wrist[kept_arr]
        ).astype(np.float32)
        right_wrist_curr_7 = np.where(
            right_wrist[kept_arr] >= INVALID_VALUE - 1, 0.0, right_wrist[kept_arr]
        ).astype(np.float32)
        actions_head = head_pose_world[kept_arr]
        left_keypoints = np.where(
            left_kps[kept_arr] >= INVALID_VALUE - 1, 0.0, left_kps[kept_arr]
        ).astype(np.float32)
        right_keypoints = np.where(
            right_kps[kept_arr] >= INVALID_VALUE - 1, 0.0, right_kps[kept_arr]
        ).astype(np.float32)

        ordered_frames = [decoded[t] for t in kept]
        del decoded

        # Preview MP4 (before JPEG encoding to avoid re-decoding)
        preview_path = task_output_dir / f"{task_id}_episode_{written:06d}.mp4"
        _save_preview_mp4(ordered_frames, preview_path, fps=fps)

        n_workers = min(img_workers, T)
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            encode_results = list(pool.map(_resize_and_encode, ordered_frames))
        del ordered_frames
        image_shape = list(encode_results[0][0])
        pre_encoded = np.array([r[1] for r in encode_results], dtype=object)
        del encode_results

        numeric_data = {
            "left.obs_ee_pose": left_curr_7,
            "right.obs_ee_pose": right_curr_7,
            "left.obs_keypoints": left_keypoints,
            "right.obs_keypoints": right_keypoints,
            "left.obs_wrist_pose": left_wrist_curr_7,
            "right.obs_wrist_pose": right_wrist_curr_7,
            "obs_head_pose": actions_head,
        }

        used_frames = [frames[t] for t in kept]
        lang_ann = _build_language_annotations(used_frames)

        episode_path = task_output_dir / f"{task_id}_episode_{written:06d}.zarr"
        meta_override = {}
        if camera_intrinsics:
            meta_override["camera_intrinsics"] = camera_intrinsics
        ZarrWriter.create_and_write(
            episode_path=episode_path,
            numeric_data=numeric_data,
            pre_encoded_image_data={
                "images.front_1": (pre_encoded, image_shape),
            },
            embodiment=robot_type,
            fps=fps,
            task=task_desc,
            annotations=lang_ann if lang_ann else None,
            enable_sharding=True,
            metadata_override=meta_override or None,
        )
        written += 1
        print(f"[{task_id}] Wrote episode {written} ({T} frames) -> {episode_path.name}")

    if os.path.exists(task_download_path):
        shutil.rmtree(task_download_path, ignore_errors=True)

    elapsed = time.perf_counter() - t_start
    print(f"[{task_id}] Done: {written} episode(s) in {elapsed:.1f}s -> {task_output_dir}")
    return {
        "episodes": written,
        "folder": folder,
        "task_desc": task_desc,
        "total_frames": n_frames,
        "output_dir": str(task_output_dir),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert Scale SFS tasks to EgoVerse Zarr episodes"
    )
    parser.add_argument("--task-ids", nargs="+", required=True, help="Scale task IDs")
    parser.add_argument("--output-dir", default="egoverse_zarr_dataset", help="Output root")
    parser.add_argument("--download-dir", default="scale_data", help="Temp download cache")
    parser.add_argument("--robot-type", default="scale_bimanual", help="Embodiment tag")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument(
        "--workers", type=int, default=1,
        help="Parallel task workers (default: 1 = sequential)",
    )
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.download_dir).mkdir(parents=True, exist_ok=True)

    img_workers = max(1, (os.cpu_count() or 4) // max(args.workers, 1))

    total_episodes = 0
    failed: list[str] = []

    if args.workers > 1:
        print(f"Running with {args.workers} parallel workers "
              f"({img_workers} image threads per worker)")
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(
                    convert_task_to_zarr,
                    task_id=tid,
                    output_dir=args.output_dir,
                    download_dir=args.download_dir,
                    robot_type=args.robot_type,
                    fps=args.fps,
                    img_workers=img_workers,
                ): tid
                for tid in args.task_ids
            }
            for future in as_completed(futures):
                tid = futures[future]
                try:
                    total_episodes += future.result()["episodes"]
                except Exception as exc:
                    print(f"[{tid}] ERROR: {exc}")
                    traceback.print_exc()
                    failed.append(tid)
    else:
        for idx, task_id in enumerate(args.task_ids, start=1):
            print(f"\n[{idx}/{len(args.task_ids)}] {task_id}")
            try:
                result = convert_task_to_zarr(
                    task_id=task_id,
                    output_dir=args.output_dir,
                    download_dir=args.download_dir,
                    robot_type=args.robot_type,
                    fps=args.fps,
                    img_workers=img_workers,
                )
                total_episodes += result["episodes"]
            except Exception as exc:
                print(f"[{task_id}] ERROR: {exc}")
                traceback.print_exc()
                failed.append(task_id)

    print(f"\n{'=' * 60}")
    print(f"Conversion complete: {len(args.task_ids)} tasks, "
          f"{len(args.task_ids) - len(failed)} ok, {len(failed)} failed")
    print(f"Episodes written: {total_episodes}")
    if failed:
        print(f"Failed: {failed}")
    print(f"Output: {Path(args.output_dir).resolve()}")
    print("=" * 60)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
