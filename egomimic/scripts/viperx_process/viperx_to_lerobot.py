"""
Convert a ViperX LeRobot dataset to EgoVerse-compatible LeRobot format.

Adds:
  - actions.joints_act  : (T, 100, 7) pre-chunked joint actions (9→7 DoF)
  - metadata.embodiment : (T, 1) int32 embodiment id

Strips shadow joints at indices 2 and 4 from 9-DoF → 7-DoF.
Updates info.json robot_type to "viperx_right_arm".

Usage:
    python viperx_to_lerobot.py \
        --input-path  /data/sybeuret/.local/huggingface/lerobot/lerobot/pick_sponge \
        --output-path /data/sybeuret/.local/huggingface/lerobot/lerobot/pick_sponge_egov \
        --repo-id     lerobot/pick_sponge_egov
"""

import argparse
import json
import logging
import shutil
from pathlib import Path

import numpy as np
import torch
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

from egomimic.rldb.embodiment.embodiment import EMBODIMENT

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


def convert(input_path: Path, output_path: Path, repo_id: str):
    # ── 1. Load source dataset ────────────────────────────────────────────────
    logger.info(f"Loading source dataset from {input_path}")
    src = LeRobotDataset(
        repo_id=repo_id,
        root=input_path,
        local_files_only=True,
    )

    # ── 2. Build new feature dict ─────────────────────────────────────────────
    src_features = dict(src.features)

    new_features = {}
    # Keep all existing non-action features unchanged
    for k, v in src_features.items():
        if k != "action":
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
        robot_type="viperx_right_arm",
        features=new_features,
        root=output_path,
    )

    embodiment_id = EMBODIMENT.VIPERX_RIGHT_ARM.value

    # ── 4. Iterate episodes ───────────────────────────────────────────────────
    num_episodes = src.num_episodes
    logger.info(f"Converting {num_episodes} episodes...")

    for ep_idx in range(num_episodes):
        logger.info(f"  Episode {ep_idx}/{num_episodes - 1}")

        # Get all frame indices for this episode
        ep_mask = src.hf_dataset["episode_index"] == ep_idx
        frame_indices = [i for i, m in enumerate(ep_mask) if m]

        # Load raw arrays for this episode
        actions_9dof = np.array(
            [src.hf_dataset[i]["action"] for i in frame_indices]
        )  # (T, 9)
        state_9dof = np.array(
            [src.hf_dataset[i]["observation.state"] for i in frame_indices]
        )  # (T, 9)

        # Process
        state_7dof, joints_act_chunk = process_episode(actions_9dof)
        state_7dof_obs = state_9dof[:, VIPERX_KEEP_INDICES]  # (T, 7)

        # T = len(frame_indices)

        for local_t, global_idx in enumerate(frame_indices):
            raw_frame = src[global_idx]

            frame = {}

            # Pass through all video/image keys
            for k, v in raw_frame.items():
                if k in (
                    "action",
                    "observation.state",
                    "timestamp",
                    "frame_index",
                    "episode_index",
                    "index",
                    "task_index",
                ):
                    continue
                frame[k] = v

            # Overwrite state with 7-DoF
            frame["observation.state"] = torch.from_numpy(state_7dof_obs[local_t])

            # Add pre-chunked actions
            frame["actions.joints_act"] = torch.from_numpy(
                joints_act_chunk[local_t]
            )  # (100, 7)

            # Add embodiment metadata
            frame["metadata.embodiment"] = torch.tensor(
                [embodiment_id], dtype=torch.int32
            )

            dst.add_frame(frame)

        # Get task string for this episode
        task_idx = int(src.hf_dataset[frame_indices[0]]["task_index"])
        task_str = src.meta.tasks[task_idx]
        dst.save_episode(task=task_str)

    dst.consolidate()
    logger.info("Done. Consolidation complete.")

    # ── 5. Verify output info.json ────────────────────────────────────────────
    info_path = output_path / "meta" / "info.json"
    with open(info_path) as f:
        info = json.load(f)
    assert (
        info["robot_type"] == "viperx_right_arm"
    ), f"robot_type mismatch: {info['robot_type']}"
    logger.info(f"Output info.json robot_type: {info['robot_type']} ✓")
    logger.info(f"Output features: {list(info['features'].keys())}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--repo-id", type=str, required=True)
    args = parser.parse_args()
    convert(args.input_path, args.output_path, args.repo_id)


if __name__ == "__main__":
    main()
