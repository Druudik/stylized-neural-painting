import math
from abc import ABC, abstractmethod

from kornia.filters import box_blur
from torch import Tensor

from painting.utils import RandomWrapper


class BrushParamSampler(ABC):

    @property
    @abstractmethod
    def brush_params_count(self) -> int:
        pass

    @abstractmethod
    def sample(self, n_brush_strokes: int, canvas: Tensor, target: Tensor, **kwargs) -> Tensor:
        """Samples n brush strokes based on the current canvas and target image.

        Args:
            n_brush_strokes (int): Number of brush strokes to generate.
            canvas (Tensor): Current RGB canvas of shape [B, 3, S, S].
            target (Tensor): Target RGB image that the algorithm is painting of shape [B, 3, S, S].

        Returns:
            Sampled params of shape [B, n_brush_strokes, brush_params_count].
        """
        pass


class TargetGuidedSampler(BrushParamSampler):
    """
    Samples n brush strokes randomly based on the loss between canvas and target. Positions with higher loss are
    more likely to be sampled. Initializes color based on the (optionally blurred) target image for faster
    optimization and sets maximum size so that each brush stroke has better gradient signal.
    """

    def __init__(
        self,
        brush_params_count: int,
        pos_param_indices: list[tuple[int, int]],
        size_param_indices: list[int],
        color_param_indices: list[tuple[int, int, int]],
        err_map_blur_kernel_size: float = 0.125,
        color_blur_kernel_size: float = 0.02,
        initial_size_of_the_brush_stroke_rand_interval: tuple[float, float] = (0.1, 0.25),
        boundary_offset: float = 0.1,
        device: str = "cpu",
        seed: int | None = None
    ):
        """Validate params and create Sampler.

        Args:
            brush_params_count (int): number of brush parameters to sample.
            pos_param_indices (List[Tuple[int, int]]): indices of the position parameters in the brush param vector.
            size_param_indices (List[int]): indices of the size parameters in the brush param vector.
            color_param_indices (List[Tuple[int, int, int]]): indices of the RGB color in the brush param vector.
            err_map_blur_kernel_size (float): normalized size of the blur kernel for the error map. In range [0, 1].
            color_blur_kernel_size (float): normalized size of the blur kernel for the target used for color sampling.
                If 0, color is taken from the unblurred target. In range [0, 1].
            initial_size_of_the_brush_stroke_rand_interval (Tuple[float, float]): initial value of size params will
                be sampled from this interval. Must be in range [0, 1].
            boundary_offset (float): Fraction of canvas size to exclude from sampling.
                Only positions in [offset, 1 - offset] range will be sampled.
                Must be in range [0, 0.5). Default: 0.1
            device (str): Device for the PyTorch tensors. Default: 'cpu'
            seed (Optional[int]): Seed for random generator.
        """
        for cpi in color_param_indices:
            if len(cpi) != 3:
                raise ValueError("Elements of colored param indices array must be tuple of size 3 (RGB)")

        if not (0 <= boundary_offset < 0.5):
            raise ValueError(f"boundary_offset must be in range [0, 0.5), got {boundary_offset}")

        if not (0.0 <= err_map_blur_kernel_size <= 1.0):
            raise ValueError(f"blur_kernel_size must be in range [0, 1], got {err_map_blur_kernel_size}")

        if not (0.0 <= color_blur_kernel_size <= 1.0):
            raise ValueError(f"color_blur_kernel_size must be in range [0, 1], got {color_blur_kernel_size}")

        self._brush_params_count = brush_params_count
        self.pos_param_size = len(pos_param_indices)
        self.x_pos_param_indices = [x for x, _ in pos_param_indices]
        self.y_pos_param_indices = [y for _, y in pos_param_indices]
        self.size_param_indices = size_param_indices
        self.color_param_indices = color_param_indices
        self.err_map_blur_kernel_normalized_size = err_map_blur_kernel_size
        self.color_blur_kernel_normalized_size = color_blur_kernel_size
        self.initial_size_of_the_brush_stroke_rand_interval = initial_size_of_the_brush_stroke_rand_interval
        self.boundary_offset = boundary_offset
        self.device = device
        self.random = RandomWrapper(seed=seed)

    @property
    def brush_params_count(self) -> int:
        return self._brush_params_count

    def sample(self, n_brush_strokes: int, canvas: Tensor, target: Tensor, **kwargs) -> Tensor:
        if canvas.shape != target.shape:
            raise ValueError("Canvas and target shape must be of the same shape.")

        if len(canvas.shape) != 4 or canvas.shape[1] != 3 or canvas.shape[-1] != canvas.shape[-2]:
            raise ValueError(f"Expected a batched input of square RGB images but got {canvas.shape}")

        batch_size = canvas.shape[0]
        canvas_size = canvas.shape[-1]

        err_map = (target - canvas).abs() / 255

        if self.err_map_blur_kernel_normalized_size > 0:
            kernel_size = math.ceil(self.err_map_blur_kernel_normalized_size * canvas_size)
            err_map = box_blur(err_map, kernel_size, border_type="constant")

        # We're making areas with high pixel loss more likely by using err_map ** 4
        err_map = err_map ** 4
        err_map = err_map.sum(dim=1, keepdim=True) / (err_map.sum(dim=[1, 2, 3], keepdim=True) + 1e-10)
        err_map = err_map.squeeze(dim=1)  # [B, canvas_size, canvas_size]

        # Adding small constant in the cases where error map is 0 everywhere.
        err_map += 1e-10

        boundary_offset_pixels = int(self.boundary_offset * canvas_size)
        if boundary_offset_pixels > 0:
            err_map[:, :boundary_offset_pixels, :] = 0
            err_map[:, -boundary_offset_pixels:, :] = 0
            err_map[:, :, :boundary_offset_pixels] = 0
            err_map[:, :, -boundary_offset_pixels:] = 0

        sampled_pos = self.random.multinomial(err_map.flatten(start_dim=1), n_brush_strokes)  # [B, n_brush_strokes]
        sampled_pos_y = sampled_pos // canvas_size
        sampled_pos_x = sampled_pos % canvas_size

        sampled_brush_strokes = self.random.rand(
            (batch_size, n_brush_strokes, self.brush_params_count),
            device=self.device
        )
        sampled_brush_strokes[:, :, self.x_pos_param_indices] = sampled_pos_x[..., None] / (canvas_size - 1)
        sampled_brush_strokes[:, :, self.y_pos_param_indices] = sampled_pos_y[..., None] / (canvas_size - 1)

        min_size, max_size = self.initial_size_of_the_brush_stroke_rand_interval
        sampled_brush_strokes[:, :, self.size_param_indices] = (
            min_size + sampled_brush_strokes[:, :, self.size_param_indices] * (max_size - min_size)
        )

        color_source = target
        if self.color_blur_kernel_normalized_size > 0:
            color_kernel_size = math.ceil(self.color_blur_kernel_normalized_size * canvas_size)
            color_source = box_blur(color_source, color_kernel_size, border_type="constant")

        # Set initial color from target image (blurred if configured) at sampled positions.
        for i in range(batch_size):
            target_colors = (color_source[i, :, sampled_pos_y[i], sampled_pos_x[i]]).permute(1, 0) / 255.0  # Shape: [n_brush_strokes, 3]
            for cpi in self.color_param_indices:
                sampled_brush_strokes[i, :, cpi] = target_colors

        return sampled_brush_strokes
