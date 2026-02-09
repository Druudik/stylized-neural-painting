import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from painting.brushes import Brush, DifferentiableBrush
from painting.loss import BrushStrokeLoss, WeightedBrushStrokeLoss
from painting.utils import RandomUniformVector, RandomVector


class Imitator:
    """
    Implements basic training logic of DifferentiableBrush with support for resuming training from the checkpoint.

    The training step consists of generating random brush parameters, using them + actual brush to generate
    targets (which might be slow non-differentiable process) and then training the model.

    The reason for implementing one training step, instead of whole training pipeline, is that we basically have
    endless amount of data (since we're generating targets at each training step), so at any point, we can just
    reset the training and continue improving the model.

    Once we have trained brush, we can use it as a differentiable neural renderer e.g. for planning by gradient descent
    or as a fast world model for the RL algorithm.
    """

    def __init__(
        self,
        model: DifferentiableBrush,
        brush: Brush,
        brush_params_generator: RandomVector | None = None,
        batch_size: int = 128,
        train_step_samples: int = 4096,
        train_step_repeats: int = 1,
        loss_fn: BrushStrokeLoss | None = None,
        initial_lr: float = 1e-3,
        optim_kwargs: dict[str, Any] | None = None,
        scheduler_milestones: list[int] | None = None,
        scheduler_gamma: float = 0.75,
        device: str = "cpu",
    ):
        """Validate params and create Imitator.

        Args:
            model (DifferentiableBrush): Brush we want to train to imitate the actual brush.
            brush (Brush): Actual brush (the one we're trying to imitate).
            brush_params_generator (RandomVector): Random generator of the brush parameters.
                It should be pickable and its state should not depend on some global variables so that training
                can resume from the checkpoint properly.
            batch_size (int): Batch size.
            train_step_samples (int): Number of random samples to generate in each training step.
            train_step_repeats (int): Number of repeats to be run on random samples that were generated in *one* training step.
                Since generating targets may be time-consuming (if the actual brush is complex enough), we might reuse
                same targets multiple times, instead of throwing them away after one use. This is similar to the
                RL with experience replay, where we might run multiple epochs on the current data.
            loss_fn (BrushStrokeLoss): Loss function that will be optimized. Defaults to WeightedBrushStrokeLoss().
            initial_lr (float): Initial learning rate for the optimizer.
            optim_kwargs (Optional): Parameters to be passed to the AdamW optimizer.
            scheduler_milestones (Optional): Scheduler milestones for MultiStepLR scheduler.
                Defaults to [1000, 1030, 1060, 1090, 1120, 1130, 1140, 1150].
            scheduler_gamma (float): Scheduler gamma.
            device (str): Device for the PyTorch tensors.
        """
        if brush_params_generator is None:
            brush_params_generator = RandomUniformVector(
                vector_size=brush.brush_params_count,
                device=device,
            )

        if brush.brush_params_count != brush_params_generator.vector_size:
            raise ValueError(
                f"Brush params count should match brush params generator vector size. "
                f"Got sizes: {brush.brush_params_count}, {brush_params_generator.vector_size}"
            )

        if model.canvas_size != brush.canvas_size:
            raise ValueError("Canvas sizes of model and brush must match.")

        if optim_kwargs is None:
            optim_kwargs = {}

        if scheduler_milestones is None:
            scheduler_milestones = [1000, 1030, 1060, 1090, 1120, 1130, 1140, 1150]

        if loss_fn is None:
            loss_fn = WeightedBrushStrokeLoss()

        self.model = model
        self.brush = brush
        self.brush_params_generator = brush_params_generator
        self.batch_size = batch_size
        self.train_step_samples = train_step_samples
        self.train_step_repeats = train_step_repeats
        self.loss_fn = loss_fn
        self.optimizer = optim.AdamW(self.model.parameters(), lr=initial_lr, **optim_kwargs)
        self.scheduler = optim.lr_scheduler.MultiStepLR(
            self.optimizer,
            milestones=scheduler_milestones,
            gamma=scheduler_gamma
        )
        self.device = device

        self.curr_train_step = 0

    def save_state(self, path: str | Path):
        checkpoint = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "brush_params_generator": self.brush_params_generator.state_dict(),
            "curr_train_step": self.curr_train_step,
        }
        torch.save(checkpoint, path)

    def load_state(self, path: str | Path):
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)

        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.scheduler.load_state_dict(checkpoint["scheduler"])
        self.brush_params_generator.load_state_dict(checkpoint["brush_params_generator"])

        self.curr_train_step = checkpoint["curr_train_step"]

    def run_train_steps(self, n_steps: int) -> list[dict[str, list[Any]]]:
        self.model.train()

        all_stats = []
        for i in range(n_steps):
            curr_stats = self.run_one_train_step()
            all_stats.append(curr_stats)
        return all_stats

    def run_one_train_step(self) -> dict[str, list[Any]]:
        """
        Runs one training step that consists of random sampling of brush parameters, generating real targets
        using the brush we want to imitate, and utilizing these targets to train the model.

        Returns:
            Dictionary of statistics such as batch_loss for the training step.
        """
        self.model.train()

        train_step_start_t = time.time()
        stats = defaultdict(list)

        with torch.no_grad():
            # We're generating train_step_samples at once to speed up generation.
            brush_params = self.brush_params_generator.sample(self.train_step_samples)
            foreground_target, alpha_mask_target = self.brush.render_brush_stroke(brush_params)

        dataset = TensorDataset(brush_params, foreground_target, alpha_mask_target)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        for epoch in range(self.train_step_repeats):
            epoch_start_t = time.time()

            for X, y_true_foreground, y_true_alpha_mask in dataloader:
                batch_start_t = time.time()

                self.optimizer.zero_grad()
                y_pred_foreground, y_pred_alpha = self.model.render_brush_stroke(X)
                loss = self.loss_fn(
                    foreground_pred=y_pred_foreground,
                    alpha_mask_pred=y_pred_alpha,
                    foreground_target=y_true_foreground,
                    alpha_mask_target=y_true_alpha_mask,
                )
                loss = loss.mean()
                loss.backward()
                self.optimizer.step()

                stats["batch_loss"].append(loss.item())
                stats["batch_duration"].append(time.time() - batch_start_t)

            stats["epoch_avg_loss"].append(np.mean(stats["batch_loss"][-len(dataloader):]))
            stats["epoch_duration"].append(time.time() - epoch_start_t)

        stats["train_step_lr"].append(self.scheduler.get_last_lr()[0])
        stats["train_step_duration"].append(time.time() - train_step_start_t)

        self.scheduler.step()
        self.curr_train_step += 1

        return stats
