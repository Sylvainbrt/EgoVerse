"""
Convert a ViperX LeRobot dataset to EgoVerse-compatible LeRobot format.

Adds:
  - actions.joints_act  : (T, 45, 7) pre-chunked joint actions (9->7 DoF)
  - metadata.embodiment : scalar int32 embodiment id per frame

Strips shadow joints at indices 2 and 4 from 9-DoF -> 7-DoF.
Updates info.json robot_type to "viperx_right_arm" or "viperx_left_arm".

Usage:
    python viperx_to_lerobot.py \
        --input-path  /data/sybeuret/.local/huggingface/lerobot/lerobot/pick_and_place \
        --output-path /data/sybeuret/.local/huggingface/lerobot/lerobot/egoverse_data_trimmed/pick_and_place_egoverse \
        --repo-id     lerobot/pick_and_place_egoverse
        --arm         right
"""

import argparse
import json
import logging
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from egomimic.rldb.embodiment.embodiment import EMBODIMENT

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ViperX 9-DoF -> 7-DoF: drop shadow joints at indices 2 and 4
VIPERX_KEEP_INDICES = [0, 1, 3, 5, 6, 7, 8]

POINT_GAP = 1
CHUNK_LENGTH = 45
RESET_SHOULDER_MAX = -70.0
RESET_ELBOW_MIN = 75.0
RESET_MIN_RUN = 12
RESET_TRIM_EXTRA_FRAMES = 30
RESET_INITIAL_TRIM_EXTRA_FRAMES = 0
RESET_MIN_KEEP_FRAMES = 30
TRIM_INITIAL_RESET = False


def get_future_points(
    arr: np.ndarray, point_gap=POINT_GAP, chunk_length=CHUNK_LENGTH
) -> np.ndarray:
    """
    arr: (T, D) -> (T, chunk_length, D)
    For each timestep t, collect chunk_length future points spaced point_gap apart.
    Pads with last point if out of bounds.
    """
    T, D = arr.shape
    indices = np.arange(chunk_length) * point_gap  # (chunk_length,)
    t_idx = np.arange(T)[:, None]  # (T, 1)
    all_idx = np.clip(t_idx + indices[None, :], 0, T - 1)  # (T, chunk_length)
    return arr[all_idx]  # (T, chunk_length, D)


def is_reset_like(
    joints_7dof: np.ndarray,
    shoulder_max: float = RESET_SHOULDER_MAX,
    elbow_min: float = RESET_ELBOW_MIN,
) -> np.ndarray:
    return (joints_7dof[:, 1] <= shoulder_max) & (joints_7dof[:, 2] >= elbow_min)


def find_initial_reset_trim_start(
    joints_7dof: np.ndarray,
    shoulder_max: float = RESET_SHOULDER_MAX,
    elbow_min: float = RESET_ELBOW_MIN,
    min_run: int = RESET_MIN_RUN,
    extra_frames: int = RESET_INITIAL_TRIM_EXTRA_FRAMES,
    min_keep_frames: int = RESET_MIN_KEEP_FRAMES,
) -> int:
    """Return the first frame to keep after removing an initial reset segment."""
    if len(joints_7dof) == 0:
        return 0

    reset_like = is_reset_like(joints_7dof, shoulder_max, elbow_min)
    if not reset_like[0]:
        return 0

    end = 0
    while end < len(joints_7dof) and reset_like[end]:
        end += 1

    run_len = end
    if run_len < min_run:
        return 0

    trim_start = min(len(joints_7dof), end + extra_frames)
    if len(joints_7dof) - trim_start < min_keep_frames:
        return 0
    return trim_start


def find_terminal_reset_trim_end(
    joints_7dof: np.ndarray,
    shoulder_max: float = RESET_SHOULDER_MAX,
    elbow_min: float = RESET_ELBOW_MIN,
    min_run: int = RESET_MIN_RUN,
    extra_frames: int = RESET_TRIM_EXTRA_FRAMES,
    min_keep_frames: int = RESET_MIN_KEEP_FRAMES,
) -> int:
    """Return the frame count to keep after removing a terminal reset segment."""
    if len(joints_7dof) == 0:
        return 0

    reset_like = is_reset_like(joints_7dof, shoulder_max, elbow_min)
    reset_indices = np.flatnonzero(reset_like)
    if len(reset_indices) == 0:
        return len(joints_7dof)

    end = int(reset_indices[-1]) + 1
    start = end - 1
    while start > 0 and reset_like[start - 1]:
        start -= 1

    run_len = end - start
    trailing_non_reset = len(joints_7dof) - end
    if run_len < min_run or trailing_non_reset > min_run:
        return len(joints_7dof)

    trim_end = max(0, start - extra_frames)
    if trim_end < min_keep_frames:
        return len(joints_7dof)
    return trim_end


def process_episode(
    actions_7dof: np.ndarray,
    point_gap: int = POINT_GAP,
    chunk_length: int = CHUNK_LENGTH,
):
    """
    actions_7dof: (T, 7)
    Returns:
        joints_7dof      : (T, 7)
        joints_act_chunk : (T, chunk_length, 7)
    """
    joints_act_chunk = get_future_points(
        actions_7dof, point_gap=point_gap, chunk_length=chunk_length
    )
    return actions_7dof, joints_act_chunk


ARM_TO_ROBOT_TYPE = {
    "right": "viperx_right_arm",
    "left": "viperx_left_arm",
}

ARM_TO_EMBODIMENT = {
    "right": EMBODIMENT.VIPERX_RIGHT_ARM.value,
    "left": EMBODIMENT.VIPERX_LEFT_ARM.value,
}


def scalar(value):
    if isinstance(value, (list, tuple)) and len(value) == 1:
        return scalar(value[0])
    if isinstance(value, np.ndarray):
        if value.shape == ():
            return value.item()
        if value.size == 1:
            return value.reshape(-1)[0].item()
    if hasattr(value, "item"):
        try:
            return value.item()
        except ValueError:
            pass
    return value


def patch_video_metadata_from_source(
    output_path: Path,
    src: LeRobotDataset,
    video_keys: list[str],
    trim_starts: list[int],
    kept_lengths: list[int],
):
    """Restore copied-video metadata and account for rows trimmed from episode starts."""
    if not video_keys:
        return

    episode_files = sorted((output_path / "meta" / "episodes").rglob("*.parquet"))
    if not episode_files:
        raise FileNotFoundError(
            f"No episode metadata parquet files found under {output_path}"
        )

    for ep_file in episode_files:
        df = pd.read_parquet(ep_file)
        for row_idx, row in df.iterrows():
            ep_idx = int(row["episode_index"])
            src_ep = src.meta.episodes[ep_idx]
            trim_start = trim_starts[ep_idx]
            kept_len = kept_lengths[ep_idx]

            for video_key in video_keys:
                chunk_col = f"videos/{video_key}/chunk_index"
                file_col = f"videos/{video_key}/file_index"
                from_col = f"videos/{video_key}/from_timestamp"
                to_col = f"videos/{video_key}/to_timestamp"

                src_from = float(scalar(src_ep.get(from_col, 0.0)))
                df.at[row_idx, chunk_col] = int(scalar(src_ep.get(chunk_col, 0)))
                df.at[row_idx, file_col] = int(scalar(src_ep.get(file_col, 0)))
                df.at[row_idx, from_col] = src_from + (trim_start / src.fps)
                df.at[row_idx, to_col] = src_from + ((trim_start + kept_len) / src.fps)

        # Keep LeRobot's video metadata dtypes intact. If these columns become
        # float, path formatting later fails on "{file_index:06d}".
        for video_key in video_keys:
            for col in (
                f"videos/{video_key}/chunk_index",
                f"videos/{video_key}/file_index",
            ):
                if col in df.columns:
                    df[col] = df[col].astype("int64")

        table = pa.Table.from_pandas(df, preserve_index=False)
        pq.write_table(table, ep_file, compression="snappy")


def convert(
    input_path: Path,
    output_path: Path,
    repo_id: str,
    arm: str,
    point_gap: int = POINT_GAP,
    chunk_length: int = CHUNK_LENGTH,
    trim_initial_reset: bool = TRIM_INITIAL_RESET,
    trim_terminal_reset: bool = True,
    reset_shoulder_max: float = RESET_SHOULDER_MAX,
    reset_elbow_min: float = RESET_ELBOW_MIN,
    reset_min_run: int = RESET_MIN_RUN,
    initial_reset_trim_extra_frames: int = RESET_INITIAL_TRIM_EXTRA_FRAMES,
    reset_trim_extra_frames: int = RESET_TRIM_EXTRA_FRAMES,
    reset_min_keep_frames: int = RESET_MIN_KEEP_FRAMES,
):
    if point_gap < 1:
        raise ValueError(f"point_gap must be >= 1, got {point_gap}")
    if chunk_length < 1:
        raise ValueError(f"chunk_length must be >= 1, got {chunk_length}")

    robot_type = ARM_TO_ROBOT_TYPE[arm]
    embodiment_id = ARM_TO_EMBODIMENT[arm]

    # ── 1. Load source dataset ────────────────────────────────────────────────
    logger.info(f"Loading source dataset from {input_path}")
    src = LeRobotDataset(
        repo_id=repo_id,
        root=input_path,
    )

    # ── 2. Build new feature dict ─────────────────────────────────────────────
    src_features = dict(src.features)

    new_features = {}
    # Keep all existing non-action features unchanged
    for k, v in src_features.items():
        if k == "action":
            continue
        if isinstance(v, dict) and v.get("dtype") == "video":
            continue
        new_features[k] = v

    # observation.state: strip to 7-DoF
    new_features["observation.state"] = {
        "dtype": "float32",
        "shape": (7,),
        "names": [
            "waist.pos",
            "shoulder.pos",
            "elbow.pos",
            "forearm_roll.pos",
            "wrist_angle.pos",
            "wrist_rotate.pos",
            "gripper.pos",
        ],
    }

    # actions.joints_act: pre-chunked (chunk_length, 7)
    new_features["actions.joints_act"] = {
        "dtype": "float32",
        "shape": (chunk_length, 7),
        "names": ["chunk_length", "action_dim"],
    }

    # metadata.embodiment
    new_features["metadata.embodiment"] = {
        "dtype": "int32",
        "shape": (1,),
        "names": ["embodiment_id"],
    }

    # ── 3. Create output dataset ──────────────────────────────────────────────
    if output_path.exists():
        shutil.rmtree(output_path)

    logger.info(f"Creating output dataset at {output_path}")
    dst = LeRobotDataset.create(
        repo_id=repo_id,
        fps=src.fps,
        robot_type=robot_type,
        features=new_features,
        root=output_path,
    )

    # LeRobot validates shape-(1,) numeric features as arrays, but serializes them
    # as scalar Values in HF datasets. Mirror the Aria converter workaround by
    # squeezing those entries right before parquet serialization.
    if not hasattr(dst, "_is_patched_for_hf_value"):
        orig_save_episode_data = dst._save_episode_data

        def safe_save_episode_data(episode_buffer):
            for key in episode_buffer:
                if key in new_features and new_features[key]["shape"] == (1,):
                    val = episode_buffer[key]
                    if (
                        isinstance(val, np.ndarray)
                        and val.ndim == 2
                        and val.shape[1] == 1
                    ):
                        episode_buffer[key] = val.squeeze(1)
                    if (
                        isinstance(val, list)
                        and len(val) > 0
                        and getattr(val[0], "shape", None) == (1,)
                    ):
                        episode_buffer[key] = [v.item() for v in val]
            return orig_save_episode_data(episode_buffer)

        dst._save_episode_data = safe_save_episode_data
        dst._is_patched_for_hf_value = True

    # ── 4. Iterate episodes ───────────────────────────────────────────────────
    num_episodes = src.num_episodes
    logger.info(f"Converting {num_episodes} episodes...")
    episode_source_lengths: list[int] = []
    episode_trim_starts: list[int] = []
    episode_terminal_drops: list[int] = []
    episode_kept_lengths: list[int] = []

    for ep_idx in range(num_episodes):
        logger.info(f"  Episode {ep_idx}/{num_episodes - 1}")

        # Get all frame indices for this episode
        ep_data = src.hf_dataset.filter(lambda x: x["episode_index"] == ep_idx)
        frame_indices = ep_data["index"]  # global frame indices

        # Load raw arrays for this episode
        actions_9dof = np.array(ep_data["action"])  # (T, 9)
        state_9dof = np.array(ep_data["observation.state"])  # (T, 9)
        original_len = len(actions_9dof)

        # Process
        action_7dof = actions_9dof[:, VIPERX_KEEP_INDICES]
        state_7dof_obs = state_9dof[:, VIPERX_KEEP_INDICES]  # (T, 7)
        trim_start = 0
        if trim_initial_reset:
            trim_start = find_initial_reset_trim_start(
                state_7dof_obs,
                shoulder_max=reset_shoulder_max,
                elbow_min=reset_elbow_min,
                min_run=reset_min_run,
                extra_frames=initial_reset_trim_extra_frames,
                min_keep_frames=reset_min_keep_frames,
            )
            if trim_start > 0:
                logger.info(
                    "    Trimming initial reset: drop %d/%d frames "
                    "(shoulder<=%.1f, elbow>=%.1f, extra=%d)",
                    trim_start,
                    len(action_7dof),
                    reset_shoulder_max,
                    reset_elbow_min,
                    initial_reset_trim_extra_frames,
                )
                action_7dof = action_7dof[trim_start:]
                state_7dof_obs = state_7dof_obs[trim_start:]
                frame_indices = frame_indices[trim_start:]

        if trim_terminal_reset:
            trim_end = find_terminal_reset_trim_end(
                action_7dof,
                shoulder_max=reset_shoulder_max,
                elbow_min=reset_elbow_min,
                min_run=reset_min_run,
                extra_frames=reset_trim_extra_frames,
                min_keep_frames=reset_min_keep_frames,
            )
            if trim_end < len(action_7dof):
                logger.info(
                    "    Trimming terminal reset: keep %d/%d frames "
                    "(shoulder<=%.1f, elbow>=%.1f, extra=%d)",
                    trim_end,
                    len(action_7dof),
                    reset_shoulder_max,
                    reset_elbow_min,
                    reset_trim_extra_frames,
                )
                action_7dof = action_7dof[:trim_end]
                state_7dof_obs = state_7dof_obs[:trim_end]
                frame_indices = frame_indices[:trim_end]

        episode_source_lengths.append(original_len)
        episode_trim_starts.append(trim_start)
        episode_terminal_drops.append(original_len - trim_start - len(action_7dof))
        episode_kept_lengths.append(len(action_7dof))

        action_7dof, joints_act_chunk = process_episode(
            action_7dof,
            point_gap=point_gap,
            chunk_length=chunk_length,
        )

        # T = len(frame_indices)
        task_idx = int(ep_data["task_index"][0])
        # tasks DataFrame is indexed by task string, task_index is a column
        task_str = src.meta.tasks[src.meta.tasks["task_index"] == task_idx].index[0]

        for local_t in range(len(frame_indices)):
            frame = {}
            frame["task"] = task_str
            frame["observation.state"] = torch.from_numpy(state_7dof_obs[local_t])
            frame["actions.joints_act"] = torch.from_numpy(
                joints_act_chunk[local_t].astype(np.float32)
            )
            frame["metadata.embodiment"] = np.array([embodiment_id], dtype=np.int32)
            dst.add_frame(frame)

        dst.save_episode()

    logger.info(
        "Trim summary: source=%d, initial_drop=%d, terminal_drop=%d, kept=%d",
        sum(episode_source_lengths),
        sum(episode_trim_starts),
        sum(episode_terminal_drops),
        sum(episode_kept_lengths),
    )

    dst.finalize()
    logger.info("Done. Finalization complete.")

    # ── 5. Copy video files directly ─────────────────────────────────────────
    logger.info("Copying video files...")
    src_videos = input_path / "videos"
    dst_videos = output_path / "videos"
    if src_videos.exists():
        if dst_videos.exists():
            shutil.rmtree(dst_videos)
        shutil.copytree(src_videos, dst_videos)
        logger.info(f"Copied videos from {src_videos} to {dst_videos}")

    # ── 6. Patch info.json to add video features back ─────────────────────────
    logger.info("Patching info.json with video features...")
    info_path = output_path / "meta" / "info.json"
    with open(info_path) as f:
        info = json.load(f)
    for k, v in src_features.items():
        if isinstance(v, dict) and v.get("dtype") == "video":
            info["features"][k] = v
    with open(info_path, "w") as f:
        json.dump(info, f, indent=4)
    logger.info(f"robot_type: {info['robot_type']} ✓")
    logger.info(f"features: {list(info['features'].keys())}")

    if src_videos.exists():
        logger.info("Restoring video metadata from source dataset...")
        video_keys = [
            k
            for k, v in src_features.items()
            if isinstance(v, dict) and v.get("dtype") == "video"
        ]
        patch_video_metadata_from_source(
            output_path,
            src,
            video_keys,
            episode_trim_starts,
            episode_kept_lengths,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--repo-id", type=str, required=True)
    parser.add_argument(
        "--arm",
        type=str,
        choices=sorted(ARM_TO_ROBOT_TYPE),
        default="right",
        help="Which ViperX arm embodiment this dataset should target.",
    )
    parser.add_argument(
        "--point-gap",
        type=int,
        default=POINT_GAP,
        help="Frame spacing between action chunk points.",
    )
    parser.add_argument(
        "--chunk-length",
        type=int,
        default=CHUNK_LENGTH,
        help="Number of points in each pre-chunked joint action.",
    )
    parser.add_argument(
        "--no-trim-terminal-reset",
        dest="trim_terminal_reset",
        action="store_false",
        help="Keep terminal reset/foldback frames instead of trimming them.",
    )
    initial_trim_group = parser.add_mutually_exclusive_group()
    initial_trim_group.add_argument(
        "--trim-initial-reset",
        dest="trim_initial_reset",
        action="store_true",
        help="Trim initial reset/folded frames.",
    )
    initial_trim_group.add_argument(
        "--no-trim-initial-reset",
        dest="trim_initial_reset",
        action="store_false",
        help="Keep initial reset/folded frames.",
    )
    parser.add_argument(
        "--reset-shoulder-max",
        type=float,
        default=RESET_SHOULDER_MAX,
        help="Shoulder threshold for reset detection.",
    )
    parser.add_argument(
        "--reset-elbow-min",
        type=float,
        default=RESET_ELBOW_MIN,
        help="Elbow threshold for reset detection.",
    )
    parser.add_argument(
        "--reset-min-run",
        type=int,
        default=RESET_MIN_RUN,
        help="Minimum contiguous reset-like frames required for trimming.",
    )
    parser.add_argument(
        "--reset-trim-extra-frames",
        type=int,
        default=RESET_TRIM_EXTRA_FRAMES,
        help="Extra frames to trim before the detected terminal reset segment.",
    )
    parser.add_argument(
        "--initial-reset-trim-extra-frames",
        type=int,
        default=RESET_INITIAL_TRIM_EXTRA_FRAMES,
        help="Extra frames to trim after the detected initial reset segment.",
    )
    parser.add_argument(
        "--reset-min-keep-frames",
        type=int,
        default=RESET_MIN_KEEP_FRAMES,
        help="Do not trim an episode below this many frames.",
    )
    parser.set_defaults(trim_initial_reset=TRIM_INITIAL_RESET)
    args = parser.parse_args()
    convert(
        args.input_path,
        args.output_path,
        args.repo_id,
        args.arm,
        point_gap=args.point_gap,
        chunk_length=args.chunk_length,
        trim_initial_reset=args.trim_initial_reset,
        trim_terminal_reset=args.trim_terminal_reset,
        reset_shoulder_max=args.reset_shoulder_max,
        reset_elbow_min=args.reset_elbow_min,
        reset_min_run=args.reset_min_run,
        initial_reset_trim_extra_frames=args.initial_reset_trim_extra_frames,
        reset_trim_extra_frames=args.reset_trim_extra_frames,
        reset_min_keep_frames=args.reset_min_keep_frames,
    )


if __name__ == "__main__":
    main()
