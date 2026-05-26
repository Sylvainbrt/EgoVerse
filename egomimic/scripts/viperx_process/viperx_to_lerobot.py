"""
Convert a ViperX LeRobot dataset to EgoVerse-compatible LeRobot format.

Adds:
  - actions.joints_act  : (T, 100, 7) pre-chunked joint actions (9→7 DoF)
  - metadata.embodiment : scalar int32 embodiment id per frame

Strips shadow joints at indices 2 and 4 from 9-DoF → 7-DoF.
Updates info.json robot_type to "viperx_right_arm" or "viperx_left_arm".

Usage:
    python viperx_to_lerobot.py \
        --input-path  /data/sybeuret/.local/huggingface/lerobot/lerobot/pick_and_place \
        --output-path /data/sybeuret/.local/huggingface/lerobot/lerobot/pick_and_place_egoverse \
        --repo-id     lerobot/pick_and_place_egoverse
        --arm         right
"""

import argparse
import json
import logging
import shutil
from pathlib import Path

import numpy as np
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from egomimic.rldb.embodiment.embodiment import EMBODIMENT
from egomimic.scripts.viperx_process.fix_episodes_metadata import fix_episodes_metadata

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ViperX 9-DoF → 7-DoF: drop shadow joints at indices 2 and 4
VIPERX_KEEP_INDICES = [0, 1, 3, 5, 6, 7, 8]

POINT_GAP = 2
CHUNK_LENGTH = 100


def get_future_points(
    arr: np.ndarray, point_gap=POINT_GAP, chunk_length=CHUNK_LENGTH
) -> np.ndarray:
    """
    arr: (T, D) → (T, chunk_length, D)
    For each timestep t, collect chunk_length future points spaced point_gap apart.
    Pads with last point if out of bounds.
    """
    T, D = arr.shape
    indices = np.arange(chunk_length) * point_gap  # (chunk_length,)
    t_idx = np.arange(T)[:, None]  # (T, 1)
    all_idx = np.clip(t_idx + indices[None, :], 0, T - 1)  # (T, chunk_length)
    return arr[all_idx]  # (T, chunk_length, D)


def process_episode(actions_9dof: np.ndarray):
    """
    actions_9dof: (T, 9)
    Returns:
        joints_7dof      : (T, 7)
        joints_act_chunk : (T, 100, 7)
    """
    joints_7dof = actions_9dof[:, VIPERX_KEEP_INDICES]  # (T, 7)
    joints_act_chunk = get_future_points(joints_7dof)  # (T, 100, 7)
    return joints_7dof, joints_act_chunk


ARM_TO_ROBOT_TYPE = {
    "right": "viperx_right_arm",
    "left": "viperx_left_arm",
}

ARM_TO_EMBODIMENT = {
    "right": EMBODIMENT.VIPERX_RIGHT_ARM.value,
    "left": EMBODIMENT.VIPERX_LEFT_ARM.value,
}


def convert(input_path: Path, output_path: Path, repo_id: str, arm: str):
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

    # actions.joints_act: pre-chunked (100, 7)
    new_features["actions.joints_act"] = {
        "dtype": "float32",
        "shape": (CHUNK_LENGTH, 7),
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

    for ep_idx in range(num_episodes):
        logger.info(f"  Episode {ep_idx}/{num_episodes - 1}")

        # Get all frame indices for this episode
        ep_data = src.hf_dataset.filter(lambda x: x["episode_index"] == ep_idx)
        frame_indices = ep_data["index"]  # global frame indices

        # Load raw arrays for this episode
        actions_9dof = np.array(ep_data["action"])  # (T, 9)
        state_9dof = np.array(ep_data["observation.state"])  # (T, 9)

        # Process
        state_7dof, joints_act_chunk = process_episode(actions_9dof)
        state_7dof_obs = state_9dof[:, VIPERX_KEEP_INDICES]  # (T, 7)

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
        logger.info("Adding missing video metadata to episodes parquet...")
        fix_episodes_metadata(output_path)


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
    args = parser.parse_args()
    convert(args.input_path, args.output_path, args.repo_id, args.arm)


if __name__ == "__main__":
    main()
