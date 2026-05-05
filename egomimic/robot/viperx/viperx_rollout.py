import select
import sys
import termios
import time
import tty
from abc import ABC, abstractmethod

import numpy as np
import torch
from robot_utils import RateLoop

from egomimic.models.denoising_policy import DenoisingPolicy
from egomimic.pl_utils.pl_model import ModelWrapper
from egomimic.rldb.embodiment.viperx import ViperX
from egomimic.utils.egomimicUtils import (
    interpolate_arr,
)

GRIPPER_WIDTH = 0.09
# Control parameters
DEFAULT_FREQUENCY = 30  # Hz
QUERY_FREQUENCY = 30
DEFAULT_RESAMPLE_LENGTH = 45

RIGHT_CAM_SERIAL = ""
LEFT_CAM_SERIAL = ""

EMBODIMENT_MAP = {
    "both": 8,
    "left": 7,
    "right": 6,
}

TEMP_DIR = "/home/robot/temp_dir"


class _KeyPoll:
    def __enter__(self):
        self.fd = sys.stdin.fileno()
        self.old = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)  # no Enter needed
        return self

    def __exit__(self, exc_type, exc, tb):
        termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old)

    def getch(self):
        if select.select([sys.stdin], [], [], 0)[0]:
            return sys.stdin.read(1)
        return None


class ViperXInterface:
    """Wrapper to interact with LeRobot API"""

    def __init__(self):
        print("Initializing LeRobot ViperX...")
        from lerobot.cameras.aria.configuration_aria import AriaCameraConfig
        from lerobot.robots import RobotConfig, make_robot_from_config

        # Configure the LeRobot instance.
        # (Update the cameras config or ports based on your exact hardware setup if needed)
        cfg = RobotConfig(
            type="viperx",
            cameras={
                "front_img_1": AriaCameraConfig()  # Or OpenCVCameraConfig depending on your setup
            },
        )
        self.robot = make_robot_from_config(cfg)
        self.robot.connect()
        print("ViperX Connected!")

    def get_obs(self):
        # Fetch the observation dict from LeRobot
        obs = self.robot.get_observation()

        # 1. Image extraction
        # Handle different possible key naming conventions in LeRobot
        img = obs.get("front_img_1")
        if img is None:
            img = obs.get("observation.images.front_img_1")

        if hasattr(img, "cpu"):
            img = img.cpu().numpy()

        # 2. Joint state extraction (7-DoF)
        state_tensor = obs.get("observation.state")
        if state_tensor is not None:
            state = state_tensor.cpu().numpy()
        else:
            # Fallback if LeRobot splits joints into individual keys
            state = np.array(
                [
                    obs["waist.pos"],
                    obs["shoulder.pos"],
                    obs["elbow.pos"],
                    obs["forearm_roll.pos"],
                    obs["wrist_angle.pos"],
                    obs["wrist_rotate.pos"],
                    obs["gripper.pos"],
                ],
                dtype=np.float32,
            )

        return {
            "front_img_1": img,  # Expected: numpy array (H, W, 3) BGR
            "joint_positions": state,  # Expected: numpy array (7,)
        }

    def set_joints(self, action):
        import torch
        # Actions from EgoVerse are 7-DoF. LeRobot ViperX physically requires 9-DoF.
        # We use the same duplicate shadow-mapping logic found in your adapter.

        act_9dof = np.zeros(9, dtype=np.float32)
        act_9dof[0] = action[0]  # waist
        act_9dof[1] = action[1]  # shoulder
        act_9dof[2] = action[1]  # shoulder_shadow
        act_9dof[3] = action[2]  # elbow
        act_9dof[4] = action[2]  # elbow_shadow
        act_9dof[5] = action[3]  # forearm
        act_9dof[6] = action[4]  # wrist_angle
        act_9dof[7] = action[5]  # wrist_rotate
        act_9dof[8] = action[6]  # gripper

        # Send action to LeRobot. Depending on LeRobot version, it accepts a torch tensor or numpy array.
        robot_action = torch.from_numpy(act_9dof)
        self.robot.send_action({"action": robot_action})

    def set_home(self):
        # LeRobot doesn't natively expose a safe .set_home().
        # For now, it's safer to leave this as pass and manually home the arm before script start.
        print("Warning: Homing is skipped. Ensure robot is mechanically safe.")
        pass

    def __del__(self):
        if hasattr(self, "robot"):
            self.robot.disconnect()


class Rollout(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def rollout_step(self, i):
        pass


class PolicyRollout(Rollout):
    def __init__(
        self,
        policy_path,
        query_frequency,
        resampled_action_len=None,
    ):
        super().__init__()
        self.policy_path = policy_path
        self.policy = ModelWrapper.load_from_checkpoint(policy_path, weights_only=False)
        self.query_frequency = query_frequency

        if getattr(self.policy.model, "diffusion", False):
            for head in self.policy.model.nets.policy.heads:
                if isinstance(
                    self.policy.model.nets.policy.heads[head], DenoisingPolicy
                ):
                    self.policy.model.nets.policy.heads[head].num_inference_steps = 10

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.policy.to(self.device)
        self.resampled_action_len = resampled_action_len
        self.actions = None

    def _downsample_chunk(self, chunk: np.ndarray, target_len: int) -> np.ndarray:
        if target_len is None or target_len <= 0 or chunk.shape[0] == target_len:
            return chunk.astype(np.float32, copy=False)
        out = interpolate_arr(chunk[None, ...], target_len)[0]
        return out.astype(np.float32, copy=False)

    def rollout_step(self, i, obs):
        if i % self.query_frequency == 0:
            start_infer_t = time.time()
            transform_list_batch = self.process_obs_for_transform_list(obs)

            # Apply your ViperX transforms
            for transform in ViperX.get_transform_list():
                transform_list_batch = transform.transform(transform_list_batch)

            for k, v in transform_list_batch.items():
                if hasattr(v, "unsqueeze"):
                    transform_list_batch[k] = v.unsqueeze(0)
                elif isinstance(v, np.ndarray):
                    transform_list_batch[k] = v[None, ...]

            batch = {"viperx_right_arm": transform_list_batch}
            processed_batch = self.policy.model.process_batch_for_training(batch)

            # Predict Joint Actions
            preds = self.policy.model.forward_eval(processed_batch)[
                "viperx_right_arm_actions_joints"
            ]
            self.actions = preds.detach().cpu().numpy().squeeze()

            if self.resampled_action_len is not None:
                self.actions = self._downsample_chunk(
                    self.actions, self.resampled_action_len
                )

            print(f"Inference time: {(time.time() - start_infer_t):.4f}s")

        act_i = i % self.query_frequency
        return self.actions[act_i]

    def process_obs_for_transform_list(self, obs):
        # front camera: obs["front_img_1"] is assumed BGR, shape [H, W, 3]
        front = torch.from_numpy(obs["front_img_1"][None, ...])
        front = front[..., [2, 1, 0]]  # BGR -> RGB
        front = front.permute(0, 3, 1, 2).to(self.device, dtype=torch.float32) / 255.0

        joints = torch.from_numpy(obs["joint_positions"]).to(
            self.device, dtype=torch.float32
        )

        data = {
            "front_img_1": front.squeeze(),
            "joint_positions": joints,
            "pad_mask": torch.ones((1, 100, 1), device=self.device, dtype=torch.bool),
            "embodiment": ["viperx_right_arm"],
            "metadata.robot_name": ["viperx_right_arm"],
        }
        return data

    def reset(self):
        self.actions = None
        self.debug_actions = None
        self.policy = ModelWrapper.load_from_checkpoint(
            self.policy_path, weights_only=False
        )
        if getattr(self.policy.model, "diffusion", False):
            for head in self.policy.model.nets.policy.heads:
                if isinstance(
                    self.policy.model.nets.policy.heads[head], DenoisingPolicy
                ):
                    self.policy.model.nets.policy.heads[head].num_inference_steps = 10


def reset_rollout(ri, policy):
    print("Resetting rollout: going home + clearing policy state")
    ri.set_home()
    if hasattr(policy, "reset"):
        policy.reset()
    if hasattr(policy, "actions"):
        policy.actions = None


def main(
    frequency,
    query_frequency=None,
    policy_path=None,
    resampled_action_len=None,
):
    # Initialize your LeRobot ViperX Wrapper
    ri = ViperXInterface()

    if policy_path is not None:
        policy = PolicyRollout(
            policy_path=policy_path,
            query_frequency=query_frequency,
            resampled_action_len=resampled_action_len,
        )
    else:
        raise ValueError("Must provide --policy-path.")

    try:
        with _KeyPoll() as kp:
            reset_rollout(ri, policy)

            while True:  # restartable
                with RateLoop(frequency=frequency, verbose=True) as loop:
                    for step_i in loop:
                        ch = kp.getch()
                        if ch == "q":
                            print("Quit requested.")
                            return
                        if ch == "r":
                            print("Restart requested.")
                            reset_rollout(ri, policy)
                            time.sleep(2.0)
                            break

                        obs = ri.get_obs()
                        actions = policy.rollout_step(step_i, obs)

                        if actions is None:
                            print(
                                "Finish rollout. Press 'r' to restart or 'q' to quit."
                            )
                            break

                        # ViperX Joint Action Step
                        ri.set_joints(actions)

    except KeyboardInterrupt:
        print("KeyboardInterrupt detected, exiting rollout.")
        return


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Rollout ViperX model.")
    parser.add_argument(
        "--frequency",
        type=float,
        default=DEFAULT_FREQUENCY,
        help="Control loop frequency in Hz",
    )
    parser.add_argument(
        "--query_frequency",
        type=int,
        default=QUERY_FREQUENCY,
        help="Frames which model does inference",
    )
    parser.add_argument("--policy-path", type=str, help="policy checkpoint path")
    parser.add_argument(
        "--resampled-action-len",
        type=int,
        default=DEFAULT_RESAMPLE_LENGTH,
        help="Resample each predicted action chunk to this length",
    )

    args = parser.parse_args()

    main(
        frequency=args.frequency,
        query_frequency=args.query_frequency,
        policy_path=args.policy_path,
        resampled_action_len=args.resampled_action_len,
    )
