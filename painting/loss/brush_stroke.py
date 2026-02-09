from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from torch import Tensor


class BrushStrokeLoss(nn.Module, ABC):

    @abstractmethod
    def forward(
        self,
        foreground_pred: Tensor,
        alpha_mask_pred: Tensor,
        foreground_target: Tensor,
        alpha_mask_target: Tensor
    ) -> Tensor:
        """Computes loss between 2 brush strokes.

        Args:
            foreground_pred (Tensor): pred RGB image of shape [B, 3, S, S].
            alpha_mask_pred (Tensor): pred alpha mask of shape [B, 1, S, S].
            foreground_target (Tensor): target RGB foreground of shape [B, 3, S, S].
            alpha_mask_target (Tensor): target alpha mask of shape [B, 1, S, S].

        Returns:
            Tensor of shape [B] with computed loss for each input pair.
        """
        pass


class MSEBrushStrokeLoss(BrushStrokeLoss):
    """Simple MSE loss between predicted and target brush strokes."""

    def forward(
        self,
        foreground_pred: Tensor,
        alpha_mask_pred: Tensor,
        foreground_target: Tensor,
        alpha_mask_target: Tensor,
    ) -> Tensor:
        """Compute MSE loss between RGBA brush stroke representations.

        Args:
            foreground_pred: Predicted RGB foreground [B, 3, S, S] in range [0, 255].
            alpha_mask_pred: Predicted alpha mask [B, 1, S, S] in range [0, 1].
            foreground_target: Target RGB foreground [B, 3, S, S] in range [0, 255].
            alpha_mask_target: Target alpha mask [B, 1, S, S] in range [0, 1].

        Returns:
            MSE loss tensor of shape [B].
        """
        foreground_pred = foreground_pred / 255
        foreground_target = foreground_target / 255

        pred = torch.cat([foreground_pred, alpha_mask_pred], dim=1)
        target = torch.cat([foreground_target, alpha_mask_target], dim=1)

        loss = (pred - target).pow(2).mean(dim=[1, 2, 3])

        return loss


class WeightedBrushStrokeLoss(BrushStrokeLoss):
    """
    This loss is a weighted combination of two components, designed to place greater emphasis on small brush strokes.
    This helps to prevent neural network from collapsing into predicting only a black background for the small brush strokes.
    """

    def __init__(
        self,
        mse_loss_weight: float = 1.0,
        alpha_loss_weight: float = 1.0,
        alpha_scale: float = 2.0,
        eps: float = 1e-6,
    ):
        """Initialize WeightedBrushStrokeLoss.

        Args:
                mse_loss_weight: Weight for the standard RGBA MSE loss component. Default: 1.0.
                alpha_loss_weight: Weight for the alpha-weighted RGB MSE loss component. Default: 1.0.
                alpha_scale: Scaling factor for alpha-based weighting. Higher values increase the
                    importance of pixels with high alpha (visible brush stroke regions). Default: 2.0.
                eps: Small constant for numerical stability when normalizing weighted errors. Default: 1e-6.
        """
        super().__init__()
        self.mse_loss_weight = mse_loss_weight
        self.alpha_loss_weight = alpha_loss_weight
        self.alpha_scale = alpha_scale
        self.eps = eps

    def forward(
        self,
        foreground_pred: Tensor,
        alpha_mask_pred: Tensor,
        foreground_target: Tensor,
        alpha_mask_target: Tensor,
    ) -> Tensor:
        """Compute weighted combination of MSE and alpha-weighted RGB MSE loss.

        Args:
            foreground_pred: Predicted RGB foreground [B, 3, S, S] in range [0, 255].
            alpha_mask_pred: Predicted alpha mask [B, 1, S, S] in range [0, 1].
            foreground_target: Target RGB foreground [B, 3, S, S] in range [0, 255].
            alpha_mask_target: Target alpha mask [B, 1, S, S] in range [0, 1].

        Returns:
            Combined loss tensor of shape [B].
        """
        foreground_pred = foreground_pred / 255.0
        foreground_target = foreground_target / 255.0

        # MSE.
        pred_rgba = torch.cat([foreground_pred, alpha_mask_pred], dim=1)
        target_rgba = torch.cat([foreground_target, alpha_mask_target], dim=1)
        mse_rgba = (pred_rgba - target_rgba).pow(2).mean(dim=[1, 2, 3])

        # Alpha weighted MSE (pixels with higher alpha have more weight).
        rgb_sq_err = (foreground_pred - foreground_target).pow(2)
        weights = 1.0 + self.alpha_scale * alpha_mask_target
        weighted_err = rgb_sq_err * weights
        alpha_weighted_rgb_mse = weighted_err.sum(dim=[1, 2, 3]) / (weights.sum(dim=[1, 2, 3]).clamp_min(self.eps))

        # Combine into final loss with proper weights.
        weight_sum = self.mse_loss_weight + self.alpha_loss_weight
        loss = (self.mse_loss_weight * mse_rgba + self.alpha_loss_weight * alpha_weighted_rgb_mse) / weight_sum
        return loss
