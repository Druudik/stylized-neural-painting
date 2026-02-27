import warnings
from collections.abc import Callable, Iterable, Sequence

import kornia
import torch
from torch import Tensor

from painting.brushes import DifferentiableBrush
from painting.loss import GatysStyleLoss, ImageLoss, PixelLoss
from painting.utils import apply_closing_to_brush_stroke, flatten, optional_wrap


class StyleTransferer:
    """Optimizes brush stroke parameters to match a style image while preserving content.

    This class takes brush parameters output from Painter.paint() and optimizes them
    to transfer style from a reference style image while maintaining structural similarity
    to the original target image.
    """

    def __init__(
        self,
        differentiable_brush: DifferentiableBrush,
        brush_pos_indices: list[tuple[int, int]],
        brush_size_indices: list[int],
        brush_color_indices: list[tuple[int, int, int]],
        lr: float = 2e-3,
        n_iters: int = 150,
        pixel_loss: ImageLoss | None = None,
        style_loss: ImageLoss | None = None,
        pixel_loss_weight: float = 1.0,
        style_loss_weight: float = 174_080,
        color_only_mode: bool = False,
        boundary_offset: float = 0.1,
        min_local_param_size: float = 0.1,
        max_local_param_size: float = 0.9,
        rendering_batch_size: int = 50,
        device: str = "cpu",
    ):
        """Validate params and initialize StyleTransferer.

        Args:
            differentiable_brush (DifferentiableBrush): Fast differentiable brush used during optimization.
            brush_pos_indices (list[tuple[int, int]]): List of (x, y) tuple position indices of the brush.
            brush_size_indices (list[int]): Size indices of the brush.
            brush_color_indices (list[tuple[int, int, int]]): Indices of RGB color parameters.
            lr (float): Learning rate for optimizer. Default: 2e-3.
            n_iters (int): Number of optimization iterations. Default: 150.
            pixel_loss (ImageLoss | None): Pixel/content loss module. Default: None.
            style_loss (ImageLoss | None): Style loss module. Default: None.
            pixel_loss_weight (float): Weight for pixel (content) loss. Default: 1.0.
            style_loss_weight (float): Weight for style loss. Default: 174_080 (this way it's equivalent to the paper's style loss
                since they do sum of the gram matrices diff instead of mean).
            color_only_mode (bool): If True, only optimize color parameters. Default: False.
            boundary_offset (float): Fraction of grid size for position clamping boundary. Default: 0.1.
            min_local_param_size (float): Minimum value for size parameters during optimization, in local grid coordinates [0, 1].
                Default: 0.1.
            max_local_param_size (float): Maximum value for size parameters during optimization, in local grid coordinates [0, 1].
                Default: 0.9.
            rendering_batch_size (int): Maximum number of strokes that are rendered at the same time and drawn on the final canvas.
                Increase this number for faster painting if you have high enough GPU memory.
            device (str): Device for PyTorch tensors. Default: 'cpu'.
        """
        if not (0 <= min_local_param_size < max_local_param_size <= 1):
            raise ValueError(
                f"min_local_param_size must be less than max_local_param_size and both in [0, 1], "
                f"got min={min_local_param_size}, max={max_local_param_size}"
            )

        self.differentiable_brush = differentiable_brush.eval()
        self.canvas_size = differentiable_brush.canvas_size
        self.brush_params_count = differentiable_brush.brush_params_count

        self.brush_x_pos_indices = [x for x, _ in brush_pos_indices]
        self.brush_y_pos_indices = [y for _, y in brush_pos_indices]
        self.brush_size_indices = brush_size_indices
        self.brush_color_indices = brush_color_indices
        self.brush_other_indices = [
            i for i in range(self.brush_params_count)
            if i not in flatten([brush_pos_indices, brush_size_indices, brush_color_indices])
        ]

        self.lr = lr
        self.n_iters = n_iters
        self.color_only_mode = color_only_mode
        self.boundary_offset = boundary_offset
        self.rendering_batch_size = rendering_batch_size
        self.device = device

        self.min_local_param_size = min_local_param_size
        self.max_local_param_size = max_local_param_size

        if pixel_loss is None:
            pixel_loss = PixelLoss(p=1, ignore_color=True)

        if style_loss is None:
            style_loss = GatysStyleLoss.create(color_only=color_only_mode, device=device)

        self.pixel_loss = pixel_loss
        self.style_loss = style_loss
        self.pixel_loss_weight = pixel_loss_weight
        self.style_loss_weight = style_loss_weight
        
        self._morph_kernel = torch.ones(3, 3, dtype=torch.float32, device=self.device)

    def transfer(
        self,
        brush_params: Tensor,
        target_image: Tensor,
        style_image: Tensor,
        n_grids_per_dim: int,
        initial_canvas: Tensor | None = None,
        apply_brush_stroke_closing: bool = True,
        iter_progress_wrapper: Callable[[Sequence[int], str], Iterable[int]] | None = None,
    ) -> Tensor:
        """Run style transfer optimization on brush parameters.

        Args:
            brush_params (Tensor): Normalized brush parameters [N, P] in full canvas coordinates.
            target_image (Tensor): Target content image [3, H, W] in range [0, 255].
            style_image (Tensor): Style reference image [3, H', W'] in range [0, 255].
            n_grids_per_dim (int): Number of grids into which target image will be split along both dimensions.
                Each input brush parameter will be assigned to exactly one grid, based on the location of its first position parameter.
                After the assignment, its size and pos params are rescaled such that position params (0, 0)
                corresponds to the top left grid corner and size param of 1 corresponds the grid's length.
                This is done to improve and speedup optimization process.
                The clamping constants: boundary_offset, min_local_param_size and max_local_param_size, are used
                on these rescaled parameters.
            initial_canvas (Tensor | None): Optional initial canvas [3, H, W] in range [0, 255].
                Default is black canvas.
            apply_brush_stroke_closing (bool): If True, applies morphological closing
                (dilation on foreground, erosion on alpha mask with 3x3 kernel) to brush strokes
                rendered by the differentiable brush during optimization.
                This was used in the original paper. It seems to help with style transfer.
            iter_progress_wrapper (Callable[[Sequence[Any], str], Iterable[Any]] | None): Optional wrapper around
                optimization cycle. This can be used for logging style transfer progress or anything else. It accepts
                sequence such as range(10) and name. It must return Iterable over this sequence.

        Returns:
            Optimized brush parameters [N, P] in full canvas coordinates.
        """
        self.differentiable_brush.eval()

        # Preprocess style image as was done in the paper. This seems to help with better style transfer.
        style_image = style_image.unsqueeze(0)
        style_image = kornia.geometry.resize(style_image, (128, 128))
        style_image = kornia.filters.blur.box_blur(style_image, 2)

        n_grids = n_grids_per_dim * n_grids_per_dim
        grid_contexts, grid_assignments = self._assign_strokes_to_grids(brush_params, n_grids_per_dim)

        # Reverse rescale to get grid-local parameters. This + parameter clamping effectively force params to stay
        # in their respective grids, which produces more appealing/realistic looking results.
        local_params = self._to_grid_local_coords(brush_params, grid_contexts)
        self._warn_if_params_would_be_clamped_before_optim_loop(local_params)
        self._clamp_params(local_params)

        local_params = local_params.detach().clone()
        local_params.requires_grad = True

        if self.color_only_mode:
            color_indices = flatten(self.brush_color_indices)
            grad_mask = torch.zeros(self.brush_params_count, dtype=torch.bool, device=self.device)
            grad_mask[color_indices] = True
        else:
            grad_mask = None

        # We're not using centered RMSprop (insync with paper's implementation).
        optimizer = torch.optim.RMSprop([local_params], lr=self.lr)

        target_grids = self._split_image_to_grids(target_image, n_grids_per_dim)

        # Initialize grids' canvases.
        if initial_canvas is not None:
            initial_grids = self._split_image_to_grids(initial_canvas, n_grids_per_dim)
        else:
            initial_grids = torch.zeros(n_grids, 3, self.canvas_size, self.canvas_size, device=self.device)

        for _ in optional_wrap(range(self.n_iters), iter_progress_wrapper, "Style transfer loop"):
            optimizer.zero_grad()

            grids = initial_grids.detach().clone()

            grids = self._draw_on_grids(
                local_params, grid_assignments, grids, n_grids, apply_brush_stroke_closing,
            )

            # Merge grids into full canvas image.
            # Note that the merged_canvas size is [1, 3, n_grids_per_dim * S, n_grids_per_dim * S].
            merged_canvas = self._merge_grids_to_image(grids, n_grids_per_dim)

            # Compute pixel loss on separate grids.
            pixel_loss = self.pixel_loss.forward(grids, target_grids).mean()

            # Compute style loss on merged canvas.
            style_loss = self.style_loss.forward(merged_canvas, style_image).mean()

            total_loss = self.pixel_loss_weight * pixel_loss + self.style_loss_weight * style_loss
            total_loss.backward()

            if self.color_only_mode:
                # Zero out gradients for non-color parameters.
                local_params.grad[:, ~grad_mask] = 0

            optimizer.step()

            self._clamp_params(local_params)

        # Rescale back to full canvas coordinates so that they can be properly drawn
        # on the final canvas.
        optimized_params = self._to_global_coords(local_params, grid_contexts)

        return optimized_params.detach()

    def _assign_strokes_to_grids(self, brush_params: Tensor, n_grids_per_dim: int) -> tuple[Tensor, Tensor]:
        """Assign grid contexts and grid indices for each brush stroke based on its first position parameter.

        Args:
            brush_params: Normalized brush parameters [N, P] in full canvas coordinates.
            n_grids_per_dim: Number of grids per dimension.

        Returns:
            Tuple of:
            - grid_contexts: Tensor [N, 3] where each row contains:
              - grid_x_start: normalized x coordinate of grid's top-left corner
              - grid_y_start: normalized y coordinate of grid's top-left corner
              - grid_size: normalized grid size
            - grid_assignments: Grid index for each stroke [N].
        """
        grid_size = 1.0 / n_grids_per_dim

        x_pos = brush_params[:, self.brush_x_pos_indices[0]]  # [N]
        y_pos = brush_params[:, self.brush_y_pos_indices[0]]  # [N]

        # Clamp to valid range to handle edge cases.
        x_pos = x_pos.clamp(0, 1 - 1e-6)
        y_pos = y_pos.clamp(0, 1 - 1e-6)

        grid_x_idx = (x_pos / grid_size).floor()
        grid_y_idx = (y_pos / grid_size).floor()

        grid_x_start = grid_x_idx * grid_size
        grid_y_start = grid_y_idx * grid_size
        grid_sizes = torch.full_like(grid_x_start, grid_size)

        grid_contexts = torch.stack([grid_x_start, grid_y_start, grid_sizes], dim=1)
        grid_assignments = (grid_y_idx * n_grids_per_dim + grid_x_idx).long()

        return grid_contexts, grid_assignments

    def _to_grid_local_coords(
        self,
        brush_params: Tensor,
        grid_contexts: Tensor,
    ) -> Tensor:
        """Convert full-canvas parameters to grid-local coordinates.

        Args:
            brush_params: Normalized brush parameters [N, P] in full canvas coordinates.
            grid_contexts: Grid context for each stroke [N, 3].

        Returns:
            Parameters in grid-local coordinates [N, P].
        """
        local_params = brush_params.clone()

        grid_x_start = grid_contexts[:, 0]  # [N]
        grid_y_start = grid_contexts[:, 1]  # [N]
        grid_size = grid_contexts[:, 2]  # [N]

        for x_idx in self.brush_x_pos_indices:
            local_params[:, x_idx] = (brush_params[:, x_idx] - grid_x_start) / grid_size

        for y_idx in self.brush_y_pos_indices:
            local_params[:, y_idx] = (brush_params[:, y_idx] - grid_y_start) / grid_size

        for size_idx in self.brush_size_indices:
            local_params[:, size_idx] = brush_params[:, size_idx] / grid_size

        return local_params

    def _to_global_coords(self, local_params: Tensor, grid_contexts: Tensor) -> Tensor:
        """Convert grid-local parameters back to full-canvas coordinates.

        Args:
            local_params: Parameters in grid-local coordinates [N, P].
            grid_contexts: Grid context for each stroke [N, 3].

        Returns:
            Parameters in full canvas coordinates [N, P].
        """
        params = local_params.clone()

        grid_x_start = grid_contexts[:, 0]
        grid_y_start = grid_contexts[:, 1]
        grid_size = grid_contexts[:, 2]

        for x_idx in self.brush_x_pos_indices:
            params[:, x_idx] = grid_x_start + local_params[:, x_idx] * grid_size

        for y_idx in self.brush_y_pos_indices:
            params[:, y_idx] = grid_y_start + local_params[:, y_idx] * grid_size

        for size_idx in self.brush_size_indices:
            params[:, size_idx] = local_params[:, size_idx] * grid_size

        return params

    def _draw_on_grids(
        self,
        params: Tensor,
        grid_assignments: Tensor,
        grids: Tensor,
        n_grids: int,
        apply_closing: bool = True,
    ) -> Tensor:
        """Draw brush strokes on their respective grids.

        Args:
            params: Grid-local brush parameters [N, P].
            grid_assignments: Grid index for each stroke [N].
            grids: Canvas grids [G, 3, canvas_size, canvas_size].
            n_grids: Total number of grids.
            apply_closing (bool): If True, applies morphological closing
                (dilation on foreground, erosion on alpha mask with 3x3 kernel) to brush strokes
                rendered by the differentiable brush during optimization. This was used in the original paper.
                It seems to help with better style transfer, even though the changes are not super noticable.

        Returns:
            Updated grids with brush strokes drawn.
        """
        brush_stroke_transform = self._brush_stroke_closing if apply_closing else None

        result_grids = []
        for grid_idx in range(n_grids):
            mask = grid_assignments == grid_idx
            if not mask.any():
                result_grids.append(grids[grid_idx])
            else:
                # Draw each stroke sequentially to maintain order.
                updated_grid = self.differentiable_brush.draw_on_single_canvas(
                    brush_params=params[mask],
                    canvas=grids[grid_idx],
                    rendering_batch_size=self.rendering_batch_size,
                    brush_stroke_transform=brush_stroke_transform,
                )
                result_grids.append(updated_grid)

        return torch.stack(result_grids)

    def _merge_grids_to_image(self, grids: Tensor, n_grids_per_dim: int) -> Tensor:
        """Merge non-overlapping grids into a single image.

        Args:
            grids: Grid patches [G, 3, S, S] where G = n_grids_per_dim^2.
            n_grids_per_dim: Number of grids per dimension.

        Returns:
            Merged image [1, 3, S*n_grids_per_dim, S*n_grids_per_dim].
        """
        s = grids.shape[-1]
        full_size = s * n_grids_per_dim
        img = torch.zeros(3, full_size, full_size, device=grids.device)

        for y_id in range(n_grids_per_dim):
            for x_id in range(n_grids_per_dim):
                grid_idx = y_id * n_grids_per_dim + x_id
                img[:, y_id * s:(y_id + 1) * s, x_id * s:(x_id + 1) * s] = grids[grid_idx]

        return img.unsqueeze(0)

    def _split_image_to_grids(self, image: Tensor, n_grids_per_dim: int) -> Tensor:
        """Split image into non-overlapping grid patches.

        The image is resized to match the grid structure (n_grids_per_dim * canvas_size).

        Args:
            image: Input image [3, H, W].
            n_grids_per_dim: Number of grids per dimension.

        Returns:
            Grid patches [G, 3, canvas_size, canvas_size].
        """
        target_size = n_grids_per_dim * self.canvas_size
        resized = kornia.geometry.resize(image.unsqueeze(0), (target_size, target_size))
        resized = resized.squeeze(0)  # [3, target_size, target_size]

        grids = []
        for y_id in range(n_grids_per_dim):
            for x_id in range(n_grids_per_dim):
                y_start = y_id * self.canvas_size
                x_start = x_id * self.canvas_size
                grid = resized[:, y_start:y_start + self.canvas_size, x_start:x_start + self.canvas_size]
                grids.append(grid)

        return torch.stack(grids).to(self.device)

    def _clamp_params(self, params: Tensor) -> None:
        """Clamp parameters to valid ranges in-place.

        Args:
            params: Parameter tensor to clamp [N, P].
        """
        with torch.no_grad():
            params.clamp_(0.0, 1.0)

            if self.boundary_offset > 0:
                pos_indices = self.brush_x_pos_indices + self.brush_y_pos_indices
                params[:, pos_indices] = params[:, pos_indices].clamp(
                    self.boundary_offset, 1.0 - self.boundary_offset
                )

            params[:, self.brush_size_indices] = params[:, self.brush_size_indices].clamp(
                self.min_local_param_size, self.max_local_param_size
            )
            
    def _brush_stroke_closing(self, foregrounds: Tensor, alpha_masks: Tensor) -> tuple[Tensor, Tensor]:
        return apply_closing_to_brush_stroke(foregrounds, alpha_masks, self._morph_kernel)

    def _warn_if_params_would_be_clamped_before_optim_loop(self, local_params: Tensor) -> None:
        problems = []
        eps = 1e-6

        # Check problems with position params.
        pos_indices = self.brush_x_pos_indices + self.brush_y_pos_indices
        pos_params = local_params[:, pos_indices]
        n_pos = ((pos_params < self.boundary_offset - eps) | (pos_params > 1.0 - self.boundary_offset + eps)).any(dim=1).sum().item()
        if n_pos > 0:
            problems.append(
                f"{n_pos} stroke(s) will be repositioned because they have some position parameters outside their grid's boundary"
            )

        # Check problems with size params.
        size_params = local_params[:, self.brush_size_indices]
        n_size = ((size_params < self.min_local_param_size - eps) | (size_params > self.max_local_param_size + eps)).any(dim=1).sum().item()
        if n_size > 0:
            problems.append(
                f"{n_size} stroke(s) will be resized because they have size parameters outside the allowed range"
            )

        if problems:
            warnings.warn(
                "Some initial brush parameters passed to StyleTransferer.transfer() are out of range due to the "
                "configured boundaries and the grid assignment logic, which assigns each brush "
                "stroke to exactly one grid for optimization stability. These parameters will be "
                "clamped before optimization, which may distort the painting in unwanted ways.\n"
                + "; ".join(problems) + ".\n"
                "It is the caller's responsibility to properly configure the StyleTransferer "
                "with: boundary_offset, min_local_param_size, max_local_param_size and n_grids_per_dim (transfer method param), "
                "to ensure brush parameters will fall within the valid range after being rescaled for their assigned grid. "
                "You can check this class (StyleTransferer) and Painter class docs for more details about the grids and rescaling logic."
            )
