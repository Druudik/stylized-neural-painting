import math
import warnings
from collections.abc import Callable, Iterable, Sequence
from typing import Any

import kornia.geometry.transform as kornia_transform
import torch
from torch import Tensor

from painting.brushes import Brush, DifferentiableBrush
from painting.loss import ImageLoss, PixelLoss
from painting.samplers import BrushParamSampler
from painting.utils import apply_closing_to_brush_stroke, flatten, optional_wrap


class Painter:
    """
    Optimizes brush strokes to resemble target image. The optimization is done via gradient descent by using
    differentiable_brush_imitator that should resemble non-differentiable brush. This class works by iteratively
    splitting image into more and more grids, running independent optimization on each one of them, aggregating
    optimized brush strokes from each grid and rescaling them so that they can be painted on the one unified output canvas.

    For the rescaling purposes, all the position and size parameters must be specified in the brush_size_indices
    and brush_pos_indices parameters. The brush strokes that don't have notion of position or size, are not supported.

    Since we're doing rescaling, size parameters must satisfy following property: when we increase brush size by factor
    of 2, the final area of the bigger brush stroke should be proportional to that as well. This doesn't hold for the brush
    strokes that have minimal size e.g. they are wide 10 pixels when their size_param == 0. The optimization would still
    work and might produce nice results though. But this property should ideally hold.

    During optimization, we use an "active set" approach where brush strokes are added incrementally and optimized
    together. Each active set starts with one stroke and grows until it reaches the target size, with joint
    optimization at each stage.
    """

    def __init__(
        self,
        brush: Brush,
        brush_pos_indices: list[tuple[int, int]],
        brush_size_indices: list[int],
        differentiable_brush_imitator: DifferentiableBrush,
        brush_param_sampler: BrushParamSampler,
        loss: ImageLoss | None = None,
        grid_batch_size: int = 50,
        rendering_batch_size: int = 5,
        optim_lr: float = 2e-3,
        boundary_offset: float = 0.1,
        min_local_param_size: float = 0.1,
        max_local_param_size: float = 0.9,
        min_brush_stroke_size: float = 0.01,
        device: str = "cpu",
    ):
        """Validate params and create Painter.

        Args:
            brush (Brush): Brush that will be used for final painting.
            brush_pos_indices (List[Tuple[int, int]]): List of (x, y) tuple position indices of the brush.
                Can't be empty list since it's needed for rescaling purposes.
            brush_size_indices (List[int]): Size indices of the brush.
                Can't be empty list since it's needed for rescaling purposes.
            differentiable_brush_imitator (DifferentiableBrush): Fast differentiable brush that's used during optimization
                for finding best set of brush strokes that resembles target image.
            brush_param_sampler (BrushParamSampler): Used for sampling initial brush strokes at the beginning of each
                optimization step.
            loss (ImageLoss | None): Loss that's being optimized. Defaults to PixelLoss(p=1)
            grid_batch_size (int): Maximum number of grids that are being optimized at the same time. Increase this number
                for faster optimization if you have high enough GPU memory.
            rendering_batch_size (int): Maximum number of strokes that are rendered at the same time and drawn on the final canvas.
                Increase this number for faster painting if you have high enough GPU memory.
            optim_lr (float): Learning rate for parameters used during optimization.
            boundary_offset (float): Position parameters will be constrained to [offset, 1-offset] during optimization.
                Must be in range [0, 0.5). Default: 0.1.
            min_local_param_size (float): Minimum value for size parameters during optimization, in local grid coordinates [0, 1].
                Default: 0.1.
            max_local_param_size (float): Maximum value for size parameters during optimization, in local grid coordinates [0, 1].
                Default: 0.9.
            min_brush_stroke_size (float): Minimum size threshold in normalized canvas coordinates [0, 1].
                After rescaling from grid to canvas, strokes with ANY size parameter above this
                threshold will NOT be discarded. Default: 0.01.
            device (str): Device for the PyTorch tensors. Default: 'cpu'
        """
        if brush.brush_params_count != differentiable_brush_imitator.brush_params_count:
            raise ValueError("Brush and brush imitator must have same brush params semantics.")

        if brush.brush_params_count != brush_param_sampler.brush_params_count:
            raise ValueError("Brush and random sampler must have same number of brush params.")

        if len(brush_pos_indices) == 0 or len(brush_size_indices) == 0:
            raise ValueError(
                "brush_pos_indices and brush_size_indices can't be empty list, since they're needed "
                "for rescaling purposes."
            )

        if not (0 <= boundary_offset < 0.5):
            raise ValueError(f"boundary_offset must be in range [0, 0.5), got {boundary_offset}")

        if not (0 <= min_local_param_size < max_local_param_size <= 1):
            raise ValueError(
                f"min_local_param_size must be less than max_local_param_size and both in [0, 1], "
                f"got min={min_local_param_size}, max={max_local_param_size}"
            )

        if min_brush_stroke_size < 0:
            raise ValueError(f"min_brush_stroke_size must be >= 0, got {min_brush_stroke_size}")

        if loss is None:
            loss = PixelLoss(p=1)

        self.brush = brush

        self.brush_x_pos_indices = [x for x, _ in brush_pos_indices]
        self.brush_y_pos_indices = [y for _, y in brush_pos_indices]
        self.brush_size_indices = brush_size_indices

        brush_param_indices = flatten([brush_pos_indices, brush_size_indices])
        if len(brush_param_indices) != len(set(brush_param_indices)):
            raise ValueError("Brush param indices have some overlap.")

        self.differentiable_brush_imitator = differentiable_brush_imitator.eval()
        self.brush_param_sampler = brush_param_sampler
        self.loss = loss
        self.grid_batch_size = grid_batch_size
        self.rendering_batch_size = rendering_batch_size
        self.boundary_offset = boundary_offset
        self.min_local_param_size = min_local_param_size
        self.max_local_param_size = max_local_param_size
        self.min_brush_stroke_size = min_brush_stroke_size

        self.optim_lr = optim_lr

        self.device = device

        self._morph_kernel = torch.ones(3, 3, dtype=torch.float32, device=self.device)

    def paint(
        self,
        target_image: Tensor,
        initial_canvas: Tensor | None = None,
        n_grids_per_dim_schedule: list[int] | None = None,
        n_strokes_per_grid_schedule: list[int] | None = None,
        active_set_size_schedule: list[int] | None = None,
        total_optim_steps_per_active_set_schedule: list[int] | None = None,
        apply_brush_stroke_closing_during_optim: bool = True,
        iter_progress_wrapper: Callable[[Sequence[Any], str], Iterable[Any]] | None = None,
    ) -> tuple[Tensor, Tensor]:
        """
        Run the optimization process starting with initial canvas. At each iteration, the current canvas and target image is split to
        n_grids_per_dim_schedule[i] ** 2 non-overlapping grids, which are optimized separately. After each optimization iteration,
        the brush strokes are painted on the canvas using real brush.

        Note that higher number of grids produces smaller brush strokes which is useful for fine-grained details.

        Also note that list schedule parameters must be of the same size.

        Args:
            target_image (Tensor): RGB image to paint of shape [3, H, W].
            initial_canvas: (Tensor | None): Starting canvas on which the target should be painted.
                Defaults to plain black canvas of size 512.
            n_grids_per_dim_schedule (List[int] | None): Defines number of independent grids along each spatial dimension
                that the canvas and target image will be split into at each optimization iteration.
                Defaults to [1, 2, 3, 4, 5], corresponding to 1, 4, 9, 16 and 25 grids in total.
            n_strokes_per_grid_schedule (List[int] | None): Number of brush strokes that will be optimized for each grid
                during each optimization iteration. Defaults to [20, 20, 20, 20, 20].
            active_set_size_schedule (List[int] | None): Maximum size of the active set at each iteration.
                Defaults to [20, 20, 20, 20, 20].
            total_optim_steps_per_active_set_schedule (List[int] | None): Total optimization steps for each active set
                at each iteration. Defaults to [500, 500, 500, 500, 500].
            apply_brush_stroke_closing_during_optim (bool): If True, applies morphological closing
                (dilation on foreground, erosion on alpha mask with 3x3 kernel) to brush strokes
                rendered by the differentiable brush during optimization.
                This was used in the original paper during painting and style transfer. It seems to help
                with better painting and style transfer in some cases.
                Note that this only affects the differentiable brush, not the final brush rendering.
            iter_progress_wrapper (Callable[[Sequence[Any], str], Iterable[Any]] | None): Optional wrapper around for cycles
                that are used by the painting process. This can be used for tracking painting progress or anything else.
                It accepts sequence such as range(10) and name. It must return Iterable over this sequence.
                There are 4 cycles which can be wrapped: progress loop optimizing finer and finer grids, batch loop over grids,
                active sets loop and inner brush param loop of one active set.

        Returns:
            Tuple of (canvas, brush_params):
            - canvas: Painted canvas of the same shape as initial_canvas
            - brush_params: Tensor of normalized brush stroke parameters of shape [N, P] that can be reused
              for painting on the canvas size of arbitrary size. N stands for number of brush strokes and
              P for number of brush parameters (brush.brush_params_count()).
        """
        if abs(target_image.shape[1] - target_image.shape[2]) / max(target_image.shape[1], target_image.shape[2]) > 0.05:
            warnings.warn(
                f"The target image of shape {target_image.shape} is not square. It will be resized to square before "
                f"optimization, which may introduce distortion. Manual cropping is recommended for the best result or "
                f"adding some background (black/white/textured...) to the both sides to match the square resolution.\n"
            )

        self.differentiable_brush_imitator.eval()

        target_img_shape = target_image.shape
        if len(target_img_shape) != 3 or target_img_shape[0] != 3:
            raise ValueError(f"Expected RGB target image (no batch dimension) but got {target_img_shape}")

        if n_grids_per_dim_schedule is None:
            n_grids_per_dim_schedule = [1, 2, 3, 4, 5]

        if n_strokes_per_grid_schedule is None:
            n_strokes_per_grid_schedule = [20, 20, 20, 20, 20]

        if active_set_size_schedule is None:
            active_set_size_schedule = [20, 20, 20, 20, 20]

        if total_optim_steps_per_active_set_schedule is None:
            total_optim_steps_per_active_set_schedule = [500, 500, 500, 500, 500]

        self._validate_optim_config(
            n_grids_per_dim_schedule=n_grids_per_dim_schedule,
            n_strokes_per_grid_schedule=n_strokes_per_grid_schedule,
            active_set_size_schedule=active_set_size_schedule,
            total_optim_steps_per_active_set_schedule=total_optim_steps_per_active_set_schedule,
        )

        if initial_canvas is None:
            canvas = torch.full((3, 512, 512), 0.0, device=self.device)
        else:
            canvas = initial_canvas.detach().clone()

        all_optimized_params = []

        loop_params = zip(
            n_grids_per_dim_schedule,
            n_strokes_per_grid_schedule,
            active_set_size_schedule,
            total_optim_steps_per_active_set_schedule,
        )
        if iter_progress_wrapper is not None:
            loop_params = iter_progress_wrapper(list(loop_params), "Main outer loop over progressive painting schedule")

        # Run progressive optimization loop (we're constantly splitting images to more and more smaller independent grids).
        for grids_per_dim, strokes_per_grid, active_set_size, total_optim_steps in loop_params:
            # Resize canvas and target image for the current number of grids.
            curr_img_size = grids_per_dim * self.differentiable_brush_imitator.canvas_size
            scaled_canvas = kornia_transform.resize(canvas, (curr_img_size, curr_img_size))
            scaled_target = kornia_transform.resize(target_image, (curr_img_size, curr_img_size))

            # Split image into multiple non-overlapping grids.
            canvas_grids, canvas_grids_info = self._split_into_grids(scaled_canvas, grids_per_dim)
            target_grids, target_grids_info = self._split_into_grids(scaled_target, grids_per_dim)

            # Optimize brush stroke params for each grid.
            grids_optimized_params = []

            grid_batches = self._batch_iter(
                tensors=[canvas_grids, target_grids, canvas_grids_info],
                batch_size=self.grid_batch_size,
            )
            if iter_progress_wrapper is not None:
                grid_batches = iter_progress_wrapper(list(grid_batches), "Loop over batches of grids with same size")

            for grid, t_grid, grid_info in grid_batches:

                # Optimize brush strokes for the current grids.
                curr_params = self._optimize_brush_params_for_grid(
                    grid, t_grid, strokes_per_grid, active_set_size, total_optim_steps,
                    apply_brush_stroke_closing_during_optim, iter_progress_wrapper,
                )

                # Rescale brush params back so we can draw them back on the original canvas.
                curr_params = self._rescale(curr_params, grid_info)
                curr_params = curr_params.flatten(start_dim=0, end_dim=1)  # [N, P]

                # Filter out strokes with size below minimum threshold
                if self.min_brush_stroke_size > 0:
                    size_params = curr_params[:, self.brush_size_indices]
                    valid_mask = (size_params >= self.min_brush_stroke_size).any(dim=1)
                    curr_params = curr_params[valid_mask]

                grids_optimized_params.append(curr_params)

            # Draw optimized brush strokes on the canvas.
            # This will be the initial canvas for the next iteration of progressive painting.
            grids_optimized_params = torch.cat(grids_optimized_params)
            canvas = self.brush.draw_on_single_canvas(
                brush_params=grids_optimized_params,
                canvas=canvas,
                rendering_batch_size=self.rendering_batch_size,
            )

            all_optimized_params.append(grids_optimized_params)

        return canvas, torch.cat(all_optimized_params)

    def _optimize_brush_params_for_grid(
        self,
        grids: Tensor,
        target_grids: Tensor,
        strokes_per_grid: int,
        active_subset_size: int,
        total_optim_steps_per_active_subset: int,
        apply_closing: bool = False,
        iter_progress_wrapper: Callable[[Sequence[int], str], Iterable[int]] | None = None,
    ) -> Tensor:
        """Runs independent optimization process for the grids in parallel, using self.differentiable_brush_imitator for drawing.

        This method uses an "active set" approach where brush strokes are added incrementally to an active set
        and optimized together. Each active set starts with one stroke and grows until it reaches active_set_size,
        with joint optimization at each stage. After each active set is fully optimized, the grids are updated
        and a new active set cycle begins.

        Note that at each stage we run fixed optim_steps = {total_optim_steps_per_active_subset // active_subset_size}.
        This means that first stroke will be optimized for whole total_optim_steps_per_active_subset, but the stroke
        that was added last to the active_set will be optimized only for optim_steps.

        Args:
            grids (Tensor): Input grids of shape [G, 3, canvas_size, canvas_size].
            target_grids (Tensor): Target grids of shape [G, 3, canvas_size, canvas_size].
            strokes_per_grid (int): Number of brush strokes to draw on each grid.
            total_optim_steps_per_active_subset (int): Total optimization steps for each active set.
            active_subset_size (int): Maximum size of each active set.
            apply_closing (bool): If True, applies morphological closing to brush strokes during optimization. Default: False.

        Returns:
            Optimized brush stroke parameters of shape [G, strokes_per_grid, P].
        """
        if grids.shape != target_grids.shape:
            raise ValueError("Shapes of the grids must match.")

        if len(grids.shape) != 4 or grids.shape[1] != 3 or grids.shape[2] != grids.shape[3]:
            raise ValueError(f"Expected a batched input of square RGB images but got {grids.shape}")

        if strokes_per_grid % active_subset_size != 0:
            raise ValueError(
                f"strokes_per_grid ({strokes_per_grid}) must be divisible by active_set_size ({active_subset_size})"
            )

        if total_optim_steps_per_active_subset % active_subset_size != 0:
            raise ValueError(
                f"total_optim_steps_per_active_set ({total_optim_steps_per_active_subset}) must be divisible by "
                f"active_set_size ({active_subset_size})"
            )

        grids = grids.detach()
        target_grids = target_grids.detach()

        # Some subtle brush stroke transforms can help during optimmization.
        brush_stroke_transform = self._brush_stroke_closing if apply_closing else None

        n_active_set_cycles = strokes_per_grid // active_subset_size
        optim_steps_per_stage = total_optim_steps_per_active_subset // active_subset_size

        all_optimized_brush_params = []

        for _ in optional_wrap(range(n_active_set_cycles), iter_progress_wrapper, "Active set loop"):
            # Active set starts empty and grows incrementally
            active_set: list[Tensor] = []

            optimizer = None

            for _ in optional_wrap(range(active_subset_size), iter_progress_wrapper, "Brush strokes optimization inner loop"):
                # Sample a new brush stroke and add to active set
                new_params = self.brush_param_sampler.sample(1, grids, target_grids)  # [B, 1, brush_params]
                new_params = new_params.squeeze(dim=1).detach().clone()
                new_params.requires_grad = True
                active_set.append(new_params)

                # Setup optimizer so that all strokes in active set are optimized together.
                # While keeping optimizer statistics (such as momentum) from previous iterations.
                if optimizer is None:
                    optimizer = torch.optim.RMSprop([new_params], lr=self.optim_lr, centered=True)
                else:
                    optimizer.add_param_group({"params": new_params})

                # Run optimization loop on the current active set.
                self._run_optim_loop(optimizer, active_set, grids, target_grids, optim_steps_per_stage, brush_stroke_transform)

            # Add final optimized parameters to the fixed list.
            for params in active_set:
                all_optimized_brush_params.append(params.detach().clone())

            # Update grids with the optimized strokes from the current final active set.
            with torch.no_grad():
                grids = self.differentiable_brush_imitator.draw_on_canvases(
                    brush_params=active_set,
                    canvases=grids,
                    brush_stroke_transform=brush_stroke_transform
                ).detach()
                # Differentiable brush imitator might produce values outside the predefined range, if it doesn't use
                # activation like tanh/sigmoid at the end (in the Stylized Neural Painting paper, they found out that no
                # activation worked better).
                grids = grids.clamp(0, 255)

        optimized_brush_params = torch.stack(all_optimized_brush_params, dim=1).detach()
        return optimized_brush_params

    def _run_optim_loop(
        self,
        optimizer: torch.optim.Optimizer,
        active_set: list[Tensor],
        grids: Tensor,
        target_grids: Tensor,
        optim_steps: int,
        brush_stroke_transform: Callable[[Tensor, Tensor], tuple[Tensor, Tensor]] | None = None,
    ) -> None:
        """Runs gradient descent optimization loop on the active set of brush strokes.

        Args:
            optimizer: The optimizer to use for gradient updates.
            active_set: List of brush parameter tensors being optimized.
            grids: Current canvas grids of shape [G, 3, canvas_size, canvas_size].
            target_grids: Target grids of shape [G, 3, canvas_size, canvas_size].
            optim_steps: Number of optimization steps to run.
            brush_stroke_transform: Transforms (foreground, alpha_mask).
        """
        for _ in range(optim_steps):
            optimizer.zero_grad()

            updated_grids = self.differentiable_brush_imitator.draw_on_canvases(
                brush_params=active_set,
                canvases=grids,
                brush_stroke_transform=brush_stroke_transform,
            )

            loss = self.loss.forward(updated_grids, target_grids)
            loss = loss.sum()
            loss.backward()

            optimizer.step()

            # Force brush strokes within configured boundaries.
            with torch.no_grad():
                for params in active_set:
                    # Clamp params to the stable range.
                    params.clamp_(0.0, 1.0)

                    # Position boundary clamping.
                    if self.boundary_offset > 0:
                        pos_indices = self.brush_x_pos_indices + self.brush_y_pos_indices
                        params[:, pos_indices] = params[:, pos_indices].clamp(
                            self.boundary_offset, 1 - self.boundary_offset
                        )

                    # Size boundary clamping.
                    params[:, self.brush_size_indices] = params[:, self.brush_size_indices].clamp(
                        self.min_local_param_size, self.max_local_param_size
                    )

    def _split_into_grids(
        self,
        img: Tensor,
        grids_per_dim: int,
    ) -> tuple[Tensor, Tensor]:
        """Split image into non overlapping grids of equal size.

        Args:
            img (Tensor): Shape of [3, S, S].
            grids_per_dim (int): number of grids per each spatial dimension (width and height).
                Total number of grids G is equal to grids_per_dim * grids_per_dim.

        Returns:
            Grids tensor of shape [G, 3, S / grids_per_dim, S / grids_per_dim] and grid info tensor of shape [G, 3]
            where 3 is normalized grid's top-left corner (x, y) and grid size and G is equal to grids_per_dim * grids_per_dim.
        """
        if len(img.shape) != 3 or img.shape[0] != 3 or img.shape[-1] != img.shape[-2]:
            raise ValueError(f"Only square RGB images are supported. Input shape: {img.shape}")

        if img.shape[-1] % grids_per_dim != 0:
            raise ValueError("Image can not be split into the grids of same size.")

        img_size = img.shape[-1]
        grid_size = img_size // grids_per_dim

        grids = []
        grids_info = []
        for h in range(grids_per_dim):
            for w in range(grids_per_dim):
                x = w * grid_size
                y = h * grid_size

                grid = img[:, y:y + grid_size, x:x + grid_size]
                grid_info = (x, y, grid_size)

                grids.append(grid)
                grids_info.append(grid_info)

        grids = torch.stack(grids)
        # For the rescaling purposes that are being done later, we're also returning normalized grid's top-left corner and grid size.
        grids_info = torch.tensor(grids_info, dtype=torch.float32, device=self.device) / img_size
        return grids, grids_info

    def _rescale(self, brush_params: Tensor, grid_contexts: Tensor) -> Tensor:
        """Rescales the brush parameters for each grid to align all grids on a unified canvas.

        This method is used for combining multiple independent grids and their brush strokes into one bigger canvas.

        Note that input and output represents *normalized* brush parameters that doesn't depend on the actual canvas size.

        Example: consider scenario where we have 4 independent grids (2x2) that we want to align and combine to the bigger canvas
        (top left grid, top right grid, bottom left and bottom right). Each grid has normalized_grid_size = 0.5.
        The brush params coordinates in the first (top-left) grid, which might be (0.5, 0.5), need to be moved to
        (0.25, 0.25) because: grid_start (0, 0) + local_pos (0.5, 0.5) * normalized_grid_size (0.5) = (0.25, 0.25).
        Similarly, size parameters are multiplied by normalized_grid_size (0.5), since the grid occupies half
        the width/height of the unified canvas.

        Args:
            brush_params (Tensor): Normalized brush parameters of shape [B, N, P].
                All brush params sharing the same batch dimension will be scaled using the same grid_context.
            grid_contexts (Tensor): Context for scaling brush parameters of shape [B, 3].
                The last dim is equal to the top left (x, y) normalized coordinates of the grid in the target unified canvas
                and normalized grid size in the target unified canvas.

        Returns:
            Rescaled brush parameters of shape [B, N, P].
        """
        if len(brush_params.shape) != 3 or brush_params.shape[-1] != self.brush.brush_params_count:
            raise ValueError(f"Invalid shape of the brush params. Shape: {brush_params.shape}")

        if len(grid_contexts.shape) != 2 or grid_contexts.shape[-1] != 3:
            raise ValueError(f"Invalid shape of the grid context. Shape: {grid_contexts.shape}")

        if brush_params.shape[0] != grid_contexts.shape[0]:
            raise ValueError(
                f"Batch dimensions of params and grid contexts must be same. "
                f"Brush params shape: {brush_params.shape}. Grid context shape: {grid_contexts.shape}"
            )

        size_param_indices = self.brush_size_indices
        x_pos_param_indices = self.brush_x_pos_indices
        y_pos_param_indices = self.brush_y_pos_indices

        brush_params = brush_params.clone()

        # Tensors are resized to [B, 1, 1] for broadcasting.
        grids_x_start = grid_contexts[:, 0].view(-1, 1, 1)
        grids_y_start = grid_contexts[:, 1].view(-1, 1, 1)
        normalized_grid_size = grid_contexts[:, 2].view(-1, 1, 1)

        brush_params[..., x_pos_param_indices] = (
            grids_x_start + brush_params[..., x_pos_param_indices] * normalized_grid_size
        )
        brush_params[..., y_pos_param_indices] = (
            grids_y_start + brush_params[..., y_pos_param_indices] * normalized_grid_size
        )
        brush_params[..., size_param_indices] = brush_params[..., size_param_indices] * normalized_grid_size

        return brush_params

    def _batch_iter(self, tensors: list[Tensor], batch_size: int = 1) -> Iterable[list[Tensor]]:
        """Iterate over multiple tensors in batches along the first dimension.

        Args:
            tensors: List of tensors to batch. All must have same first dimension size.
            batch_size: Number of elements per batch. Default: 1.

        Yields:
            List of tensor slices, one per input tensor, each of shape [batch_size, ...].
        """
        shapes = [t.shape[0] for t in tensors]
        if len(set(shapes)) != 1:
            raise ValueError(f"All inputs must have same length of the first dimension. Got {shapes}")

        n_inputs = tensors[0].shape[0]

        for i in range(math.ceil(n_inputs / batch_size)):
            curr_start = i * batch_size
            yield [t[curr_start:curr_start + batch_size] for t in tensors]

    def _validate_optim_config(
        self,
        n_grids_per_dim_schedule: list[int],
        n_strokes_per_grid_schedule: list[int],
        active_set_size_schedule: list[int],
        total_optim_steps_per_active_set_schedule: list[int],
    ):
        schedule_lengths = [
            len(n_grids_per_dim_schedule),
            len(n_strokes_per_grid_schedule),
            len(active_set_size_schedule),
            len(total_optim_steps_per_active_set_schedule),
        ]
        if len(set(schedule_lengths)) != 1:
            raise ValueError(
                f"All schedules must have the same length. Got lengths: "
                f"n_grids_per_dim={len(n_grids_per_dim_schedule)}, "
                f"n_strokes_per_grid={len(n_strokes_per_grid_schedule)}, "
                f"active_set_size={len(active_set_size_schedule)}, "
                f"total_optim_steps_per_active_set={len(total_optim_steps_per_active_set_schedule)}"
            )

        # Validate divisibility constraints
        for i, (n_strokes, active_size, total_steps) in enumerate(zip(
            n_strokes_per_grid_schedule,
            active_set_size_schedule,
            total_optim_steps_per_active_set_schedule,
        )):
            if n_strokes % active_size != 0:
                raise ValueError(
                    f"n_strokes_per_grid_schedule[{i}] ({n_strokes}) must be divisible by "
                    f"active_set_size_schedule[{i}] ({active_size})"
                )
            if total_steps % active_size != 0:
                raise ValueError(
                    f"total_optim_steps_per_active_set_schedule[{i}] ({total_steps}) must be divisible by "
                    f"active_set_size_schedule[{i}] ({active_size})"
                )

    def _brush_stroke_closing(self, foregrounds: Tensor, alpha_masks: Tensor) -> tuple[Tensor, Tensor]:
        return apply_closing_to_brush_stroke(foregrounds, alpha_masks, self._morph_kernel)
