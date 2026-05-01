from egomimic.rldb.embodiment.embodiment import Embodiment


class ViperX(Embodiment):
    """
    ViperX Robot Arm Embodiment Class.
    Handles data transformations and key mappings for the 7-DoF ViperX dataset.
    """

    @classmethod
    def get_keymap(cls) -> dict:
        """
        Maps raw dataset keys (LeRobot) to EgoVerse internal processing keys.
        """
        return {
            "observation.images.front_img_1": "front_img_1",
            "observation.images.right_wrist_img": "right_wrist_img",
            "observation.state": "joint_positions",
            "actions.joints_act": "actions_joints",
            "metadata.embodiment": "embodiment",
        }

    @classmethod
    def get_transform_list(cls) -> list:
        """
        Defines the sequence of transformations applied to the data.
        Since the 9->7 DoF action/state stripping is handled in viperx_to_lerobot.py,
        we primarily just map keys and ensure image formats here.
        """
        return [
            # Rename raw keys to internal target names based on get_keymap
            {
                "_target_": "egomimic.rldb.transforms.transform_units.RenameKey",
                "key_map": cls.get_keymap(),
            },
            # Ensure images are in [B, C, H, W] format and float32 [0, 1] if needed
            {
                "_target_": "egomimic.rldb.transforms.transform_units.ImageFormatTransform",
                "keys": ["front_img_1", "right_wrist_img"],
            },
        ]
