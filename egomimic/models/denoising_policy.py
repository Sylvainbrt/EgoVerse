import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

from egomimic.models.denoising_nets import ConditionalUnet1D

class DenoisingPolicy(nn.Module):
    """
    Template class for a diffusion-based policy head.

    Args:
        model (ConditionalUnet1D): The model used for prediction.
        action_horizon (int): The number of time steps in the action horizon.
        infer_ac_dims (dict): Dictionary mapping embodiment names to output dimensions.
        num_inference_steps (int, optional): The number of diffusion inference steps.
        **kwargs: Optional settings like padding, pooling, robot type, etc.
    """

    def __init__(
        self,
        model: ConditionalUnet1D,
        action_horizon: int,
        infer_ac_dims: dict,
        num_inference_steps: int = None,
        **kwargs,
    ):
        super().__init__()

        self.model = model
        self.action_horizon = action_horizon
        self.infer_ac_dims = infer_ac_dims
        self.num_inference_steps = num_inference_steps

        self.padding = kwargs.get("padding", None)
        self.pooling = kwargs.get("pooling", None)
        self.human_6dof = kwargs.get("human_6dof", False)
        self.robot_cartesian = kwargs.get("robot_cartesian", False)
        self.model_type = kwargs.get("model_type", None)
        self.robot_domain = kwargs.get("robot_domain", None)

        if not infer_ac_dims:
            raise ValueError("infer_ac_dims must be a non-empty dict")

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                print(f"[warn] {name} has requires_grad=False")

        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"[{self.__class__.__name__}] Total trainable parameters: {total_params / 1e6:.2f}M")

    def preprocess_sampling(self, global_cond, embodiment_name, generator=None):
        if self.pooling == "mean":
            global_cond = global_cond.mean(dim=1)
        elif self.pooling == "flatten":
            global_cond = global_cond.reshape(global_cond.shape[0], -1)

        noise = torch.randn(
            (len(global_cond), self.action_horizon, self.infer_ac_dims[embodiment_name]),
            dtype=global_cond.dtype,
            device=global_cond.device,
            generator=generator,
        )
        return noise, global_cond

    def inference(self, noise, global_cond, generator=None) -> torch.Tensor:
        """
        To be implemented in subclass: predict actions from noise and conditioning.
        """
        raise NotImplementedError

    def sample_action(self, global_cond, embodiment_name, generator=None):
        noise, global_cond = self.preprocess_sampling(global_cond, embodiment_name, generator)
        return self.inference(noise, global_cond, generator)

    def forward(self, global_cond):
        cond, embodiment = global_cond
        return self.sample_action(cond, embodiment)

    def predict(self, actions, global_cond) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        To be implemented in subclass: returns (prediction, target) given action input and conditioning.
        """
        raise NotImplementedError

    def loss_fn(self, pred, target):
        """
        Computes loss depending on target shape and flags.
        """
        if target.shape[-1] == 6:
            loss = F.smooth_l1_loss(pred[..., :3], target[..., :3])
            if self.human_6dof:
                loss += F.smooth_l1_loss(pred[..., 3:6], target[..., 3:6])
            return loss

        elif target.shape[-1] == 12:
            loss = (
                F.smooth_l1_loss(pred[..., :3], target[..., :3]) +
                F.smooth_l1_loss(pred[..., 6:9], target[..., 6:9])
            )
            if self.human_6dof:
                loss += (
                    F.smooth_l1_loss(pred[..., 3:6], target[..., 3:6]) +
                    F.smooth_l1_loss(pred[..., 9:12], target[..., 9:12])
                )
            return loss

        elif self.robot_cartesian:
            xyz1, ypr1, grip1 = pred[..., :3], pred[..., 3:6], pred[..., 6]
            xyz1_gt, ypr1_gt, grip1_gt = target[..., :3], target[..., 3:6], target[..., 6]
            loss = (
                F.smooth_l1_loss(xyz1, xyz1_gt) +
                0.5 * F.mse_loss(ypr1, ypr1_gt) +
                F.smooth_l1_loss(grip1, grip1_gt)
            )
            if target.shape[-1] == 14:
                xyz2, ypr2, grip2 = pred[..., 7:10], pred[..., 10:13], pred[..., 13]
                xyz2_gt, ypr2_gt, grip2_gt = target[..., 7:10], target[..., 10:13], target[..., 13]
                loss += (
                    F.smooth_l1_loss(xyz2, xyz2_gt) +
                    0.5 * F.mse_loss(ypr2, ypr2_gt) +
                    F.smooth_l1_loss(grip2, grip2_gt)
                )
            return loss

        return F.smooth_l1_loss(pred, target)

    def preprocess_compute_loss(self, global_cond, data):
        if self.pooling == "mean":
            global_cond = global_cond.mean(dim=1)
        elif self.pooling == "flatten":
            global_cond = global_cond.reshape(global_cond.shape[0], -1)

        actions = data["action"].reshape(len(global_cond), self.action_horizon, -1)

        if self.padding is not None:
            if actions.shape[-1] in {6, 12}:
                pad_shape = (*actions.shape[:-1], 1)
                padding = (
                    torch.randn(pad_shape, device=actions.device)
                    if self.padding == "gaussian"
                    else torch.zeros(pad_shape, device=actions.device)
                )
                if actions.shape[-1] == 6:
                    actions = torch.cat((actions, padding), dim=-1)
                else:  # 12
                    actions = torch.cat((actions[..., :6], padding, actions[..., 6:], padding), dim=-1)

        return actions, global_cond

    def compute_loss(self, global_cond, data):
        actions, global_cond = self.preprocess_compute_loss(global_cond, data)
        pred, target = self.predict(actions, global_cond)
        return self.loss_fn(pred, target)
