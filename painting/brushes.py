from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence

import kornia
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from painting.networks import (
    ModulatedPixelShuffleNet,
    ModulatedPixelShuffleNetLight,
    PixelShuffleNet,
    PixelShuffleNetLight,
)
from painting.utils import RandomWrapper, ScaleAndTranslate, draw_non_convex_polygon


class Brush(ABC):
    """
    Brush that's able to render brush stroke's foreground and alpha mask based on the given brush stroke parameters.
    The given parameters must be normalized into the range [0, 1]. The convention for position parameters should be:
    [0, 0] top-left corner and [1, 1] bottom right.
    """

    @property
    @abstractmethod
    def brush_params_count(self) -> int:
        pass

    @property
    @abstractmethod
    def canvas_size(self) -> int:
        pass

    def draw_on_canvases(
        self,
        brush_params: Tensor | list[Tensor],
        canvases: Tensor,
        brush_stroke_transform: Callable[[Tensor, Tensor], tuple[Tensor, Tensor]] | None = None,
        **kwargs,
    ) -> Tensor:
        """Draws brush stroke on a given canvas using classic alpha blending.

        Method draws each brush stroke on a canvas matching batch dimension.

        Args:
            brush_params (Tensor | List[Tensor]): Normalized batched brush parameters (in a continuous range of [0, 1])
                of shape [B, N] or a list of brush parameters of the same shape that will be drawn iteratively on their
                respective canvases.
            canvases (Tensor): Batched RGB canvas of shape [B, 3, canvas_size, canvas_size].
            brush_stroke_transform (Callable | None): Optional function that transforms (foreground, alpha_mask)
                after rendering but before alpha blending. Receives batched tensors of shape
                [B, 3, S, S] and [B, 1, S, S], returns transformed tensors of the same shapes.
                Use cases can include: simulating random effects (noise, distortion), transforming brush strokes for better
                gradient signal etc. Default: None (no transformation).

        Returns:
            Tensor: Canvas that is updated with the rendered brush stroke.
        """
        if len(canvases.shape) != 4 or canvases.shape[1:] != (3, self.canvas_size, self.canvas_size):
            raise ValueError(f"Invalid shape of the canvas {canvases.shape}")

        if isinstance(brush_params, Tensor):
            brush_params = [brush_params]

        for bp in brush_params:
            foregrounds, alpha_masks = self.render_brush_stroke(bp, **kwargs)
            if brush_stroke_transform is not None:
                foregrounds, alpha_masks = brush_stroke_transform(foregrounds, alpha_masks)
            canvases = foregrounds * alpha_masks + canvases * (1 - alpha_masks)

        return canvases

    def draw_on_single_canvas(
        self,
        brush_params: Tensor | list[Tensor],
        canvas: Tensor,
        rendering_batch_size: int = 1,
        brush_stroke_transform: Callable[[Tensor, Tensor], tuple[Tensor, Tensor]] | None = None,
        **kwargs,
    ) -> Tensor:
        """Draws brush stroke on a given canvas one by one using classic alpha blending.

        Args:
            brush_params (Tensor | List[Tensor]): Normalized (in a continuous range of [0, 1]) brush parameters
                of shape [P] or [N, P] or list of brush parameters of shape [P] that will be drawn on the same canvas one by one.
            canvas (Tensor): RGB canvas of shape [3, canvas_size, canvas_size].
            rendering_batch_size (int): Number of brush strokes to render in a single batch. Defaults to 1.
            brush_stroke_transform (Callable | None): Optional function that transforms (foreground, alpha_mask)
                after rendering but before alpha blending. Receives batched tensors of shape
                [B, 3, S, S] and [B, 1, S, S], returns transformed tensors of the same shapes.
                Use cases can include: simulating random effects (noise, distortion), transforming brush strokes for better
                gradient signal etc. Default: None (no transformation).

        Returns:
            Tensor: Canvas that is updated with the rendered brush strokes.
        """
        if canvas.shape != (3, self.canvas_size, self.canvas_size):
            raise ValueError(f"Invalid shape of the canvas {canvas.shape}")

        if not isinstance(brush_params, Tensor):
            brush_params = torch.stack(brush_params, dim=0)  # [N, P]
        elif len(brush_params.shape) == 1:
            brush_params = brush_params.unsqueeze(0)
        elif len(brush_params.shape) != 2:
            raise ValueError("Input must be either Tensor of shape [P] or [N, P], or list of tensors of shape [P].")

        for i in range(0, brush_params.shape[0], rendering_batch_size):
            # Render brush params in batches for more efficient painting.
            batch = brush_params[i:i + rendering_batch_size]
            foregrounds, alpha_masks = self.render_brush_stroke(batch, **kwargs)

            if brush_stroke_transform is not None:
                foregrounds, alpha_masks = brush_stroke_transform(foregrounds, alpha_masks)

            for foreground, alpha_mask in zip(foregrounds.unbind(0), alpha_masks.unbind(0)):
                canvas = foreground * alpha_mask + canvas * (1 - alpha_mask)

        return canvas

    @abstractmethod
    def render_brush_stroke(self, brush_params: Tensor, **kwargs) -> tuple[Tensor, Tensor]:
        """Renders brush stroke on a black canvas.

        This method can be useful when we want to do custom alpha blending of the (foreground, alpha mask) with some background.

        Args:
            brush_params (Tensor): Batched brush parameters of shape [B, N].

        Returns:
            tuple[Tensor, Tensor]: A tuple containing:
                - Brush stroke rendered on the black canvas of shape [B, 3, canvas_size, canvas_size]. Values are in range [0, 255]
                - Alpha mask of shape [B, 1, canvas_size, canvas_size]. Values are in range [0, 1]
        """
        pass


class WatercolorBrush(Brush):
    """
    Brush that resembles leaking watercolor brush.

    Inspired by the https://www.tylerxhobbs.com/words/a-guide-to-simulating-watercolor-paint-with-generative-art

    Brief explanation of the algorithm so that args make more sense:
    The first step of this brush is to initialize its textures. Each texture is created by stacking n layers onto
    each other using classic alpha blending. Layer is created by drawing deformed polygon on the canvas and filling
    it with ones. Deformed polygons are created by recursively adding points to the middle of the edges starting with regular
    polygons with n initial polygon sides. Each such a point is then shifted in a random direction, which creates
    irregular look of the polygon. The noise by which it's shifted is controled by the variance parameters and is decayed
    with increasing depth. To create more complex texture shapes, the algorithm first deforms the initial regular polygon
    with a significant depth to generate "base deformed polygons" (one for each texture). These base polygons are then
    deformed again with depth=1 for each layer. As a result, these layers share a substantial portion
    of their area but also have slight variations along the edges, producing an intriguing, irregular look.

    The brush stroke is generated along its trajectory by randomly sampling n of these textures (as specified by "iters" param),
    scaling them to the proper size (as specified by the brush stroke radius) and moving them to the proper location.

    This basically resembles tapping of the watercolor brush n times along the stroke's trajectory.
    """

    # This field stores pregenerated watercolor-like textures. To create a brush stroke, textures are randomly sampled,
    # transformed, and layered along the stroke path. The texture shape is [N, 2, S, S], where N is the
    # number of textures, S is the canvas size, and 2 represents color and alpha mask which are combined when drawing
    # texture on some existing canvas of shape [3, S, S]
    # Each texture is created by stacking L random layers onto each other via alpha blending. One layer is
    # created by generating randomly deformed polygon and then filling it with alpha 1.0.
    # Since textures can overlap along the brush stroke trajectory, we also keep alpha blending mask (second channel)
    # that basically says "how many times" was certain pixel touched when we were stacking L layers during the texture creation.
    # This allows us to do proper alpha blending of the existing canvas and textures, which would be equivalent to alpha blending
    # L * T layers, one by one, along the whole trajectory of size T.
    _textures: Tensor  # Shape [N, 2, S, S]

    def __init__(
        self,
        canvas_size: int,
        *,
        iters: int = 100,
        n_textures: int = 40,
        min_radius_fraction: float = 0.002,
        max_radius_fraction: float = 0.2,
        n_initial_polygon_sides: int = 7,
        n_layers: int = 40,
        depth: int = 5,
        variance_decay: float = 0.4,
        radius_to_base_variance_fraction: float = 0.225,
        radius_to_final_variance_fraction: float = 0.015,
        layer_blending_alpha: float = 0.04,
        fading_strength: float = 0.3,
        alpha_mask_scale: float = 1.0,
        default_interpolation_mode: str = "nearest",
        apply_closing: bool = False,
        device: str = "cpu",
        seed: int | None = 1
    ):
        """Validate params and create WatercolorBrush.

        Args:
            canvas_size (int): Size of the canvas.
            iters (int): Number of textures that are drawn on the canvas along the brush stroke trajectory.
                Note that stacking more textures onto each other can result in less transparent + more color saturated
                brush strokes. Also note that too few iters can produce brush strokes with empty space between stacked textures,
                especially if the brush stroke is long with small radius.
            n_textures (int): Number of textures to create.
                These are randomly sampled and stacked onto each other along brush stroke trajectory.
                Higher number will produce less repeating look. Nonetheless, even small number of textures such as 100
                provide enough diversity.
            min_radius_fraction (float): Min normalized radius of the polygon (in a range of [0, 1]).
                This number is multiplied by canvas_size to obtain min radius of the polygon.
                Used during denormalization of the brush stroke parameters.
            max_radius_fraction (float): Max normalized radius of the polygon (in a range of [0, 1]).
                This number is multiplied by canvas_size to obtain max radius of the polygon.
                Used during denormalization of the brush stroke parameters.
            n_initial_polygon_sides (int): Initial number of polygon sides that is recursively deformed, filled
                with color and stacked onto each other to create texture. Higher number produces more fine-grained look.
            n_layers (int): Number of polygons that will be stacked onto each other after deformation to create one texture.
                Higher number produces more fine-grained and uniform look.
            depth (int): Control polygon deformation depth.
                At each step, new vertices are added along each edge that produce some random deformation.
                Higher number produces more irregular polygons (layers) which results in a less uniform brush stroke look.
            variance_decay (float): Decay rate of polygon deformation at each depth.
            radius_to_base_variance_fraction (float): Defines relationship between polygon radius and variance of
                the base deformed polygons. Higher values produce less uniform look similar to overly watery look
                (lot of leaking to the sides).
            radius_to_final_variance_fraction (float): Defines relationship between polygon radius and variance of
                the final deformed polygons. Final polygons are produced by deforming base polygons with depth=1.
                Higher values produce less uniform look with a lot of thin spikes on the sides.
            layer_blending_alpha (float): Alpha blending factor of the layers in a range (0, 1].
            fading_strength (float): Factor in range [0, 1] by which brush stroke is faded along its trajectory.
                0 means no fading, 1 means maximum fading.
            alpha_mask_scale (float): Specifies how should be the final alpha mask scaled.
                Lower values make brush stroke more transparent.
            default_interpolation_mode (str): Interpolation mode to be used during texture transformation (scaling).
                Possible values are 'bilinear' or 'nearest'. Default: 'nearest'
            apply_closing (bool): If True, applies dilation to foreground and erosion to alpha mask
                using a 2x2 kernel. This helps to generate smoother version of stroke with prettier
                overlapping brush strokes. Default: True
            device (str): Device for the PyTorch tensors. Default: 'cpu'
            seed (int): Seed for random generator. It might be important to set it when training
                differentiable brush to mimic it, because we want matching behavior during painting. Default: 1
        """
        if max_radius_fraction < min_radius_fraction:
            raise ValueError(
                f"Max radius size must be bigger than the min radius size. "
                f"Got: max = {max_radius_fraction:.4f}, min = {min_radius_fraction:.4f}"
            )

        self._canvas_size = canvas_size
        self.iters = iters
        self.min_radius = min_radius_fraction * canvas_size
        self.max_radius = max_radius_fraction * canvas_size
        self.n_textures = n_textures
        self.n_initial_polygon_sides = n_initial_polygon_sides
        self.n_layers = n_layers
        self.depth = depth
        self.variance_decay = variance_decay
        self.radius_to_base_variance_fraction = radius_to_base_variance_fraction
        self.radius_to_final_variance_fraction = radius_to_final_variance_fraction
        self.layer_blending_alpha = layer_blending_alpha
        self.fading_strength = fading_strength
        self.alpha_mask_scale = alpha_mask_scale
        self.default_interpolation_mode = default_interpolation_mode
        self.apply_closing = apply_closing

        # Texture radius should be big enough to keep as much detail as possible when resizing + it should also fit on the
        # canvas as a whole.
        self._texture_radius = canvas_size * 0.2

        self.device = device
        self.seed = seed
        self.random = RandomWrapper(seed=seed)

        self._const_one = torch.tensor([1.0], device=device)

        self._canvas_center = torch.tensor([canvas_size / 2, canvas_size / 2], dtype=torch.float32, device=device)

        self._scale_and_translate = ScaleAndTranslate(canvas_size, device)

        self._morph_kernel = torch.ones(2, 2, dtype=torch.float32, device=device)

        self._initialize_textures()

    def _initialize_textures(self):
        """Initialize watercolor textures by stacking deformed polygons.

        Creates n_textures unique textures, each composed of n_layers stacked deformed polygons.
        Each texture stores both a color mask and an alpha blending mask for proper compositing.
        """
        base_deform_variance = self._texture_radius * self.radius_to_base_variance_fraction
        final_deform_variance = self._texture_radius * self.radius_to_final_variance_fraction

        regular_polygons = self._create_regular_polygon(self._texture_radius, self.n_initial_polygon_sides)
        regular_polygons = regular_polygons.expand(self.n_textures, -1, -1)

        # We're creating deformed polygons with maximum possbile radius and centering them on the canvas, so that
        # we're able to do proper upscaling/downscaling without loosing too much detail.
        base_deformed_polygons = self._deform_polygon(regular_polygons, base_deform_variance, self.depth)
        base_deformed_polygons += self.canvas_size / 2

        texture_shape = (self.n_textures, 1, self.canvas_size, self.canvas_size)
        texture_color_mask = torch.zeros(texture_shape, dtype=torch.float32, device=self.device)
        texture_touched_counts = torch.zeros_like(texture_color_mask)

        for _ in range(self.n_layers):

            final_polygons = self._deform_polygon(base_deformed_polygons, final_deform_variance, 1)

            layer_color_mask = torch.zeros(texture_shape, dtype=torch.float32, device=self.device)

            layer_color_mask = draw_non_convex_polygon(
                images=layer_color_mask,
                polygons=final_polygons,
                colors=self._const_one,
            )

            texture_touched_counts += layer_color_mask
            alpha_blending_mask = (layer_color_mask > 0) * self.layer_blending_alpha
            texture_color_mask = layer_color_mask * alpha_blending_mask + texture_color_mask * (1 - alpha_blending_mask)

        self._textures = torch.cat(
            tensors=[
                texture_color_mask,
                1 - torch.pow(1 - self.layer_blending_alpha, texture_touched_counts),
            ],
            dim=1
        )

    @property
    def brush_params_count(self) -> int:
        return 10

    @property
    def canvas_size(self) -> int:
        return self._canvas_size

    def render_brush_stroke(self, brush_params: Tensor, **kwargs) -> tuple[Tensor, Tensor]:
        """Renders normalized watercolor brush strokes on a black canvas.

        For more info see parent method Brush.render_brush_stroke

        Args:
            brush_params (Tensor): Batched normalized brush parameters of shape [B, 10] in a range of [0, 1].
                Order of the params is [x0, y0, x1, y1, x2, y2, radius, r, g, b] where (x0, y0) is start,
                (x1, y1) is middle control point, and (x2, y2) is end point of a quadratic Bezier curve.

        Keyword Args:
            interpolation_mode (Optional[str]):
                One of the interpolation modes to use when upscaling/downscaling the pregenerated textures.

                Possbile values are 'bilinear' or 'nearest'. Default 'nearest'.
        """
        if brush_params.shape[1] != self.brush_params_count:
            raise ValueError(f"Invalid shape of brush parameters. Expected [B, 10] but found {brush_params.shape}")

        mode = kwargs.get("interpolation_mode", self.default_interpolation_mode)

        batch_size = brush_params.shape[0]

        brush_params = self._denormalize(brush_params)
        x0, y0, x1, y1, x2, y2, radius, r, g, b = brush_params.T
        base_color = torch.stack([r, g, b], dim=1)

        foreground_color_mask = torch.zeros((batch_size, 3, self.canvas_size, self.canvas_size), dtype=torch.float32,
                                            device=self.device)
        foreground_alpha_mask = torch.zeros((batch_size, 1, self.canvas_size, self.canvas_size), dtype=torch.float32,
                                            device=self.device)

        for i in range(self.iters):
            t = i / self.iters
            # Quadratic Bezier: B(t) = (1-t)^2 * P0 + 2(1-t)t * P1 + t^2 * P2
            one_minus_t = (1 - t)
            x = one_minus_t * one_minus_t * x0 + 2 * one_minus_t * t * x1 + t * t * x2
            y = one_minus_t * one_minus_t * y0 + 2 * one_minus_t * t * y1 + t * t * y2

            inv_fade_start = 0.5 + 0.5 * (1 - self.fading_strength)
            inv_fade_start = np.clip(inv_fade_start, 0.5, 1.0)
            inv_fade = inv_fade_start + (1 - inv_fade_start) * t

            idxs = self.random.rand_int(0, self.n_textures, (batch_size,), device=self.device)
            texture = self._textures[idxs]  # [B, 2, C, C]

            # Scale and move texture to proper location. Note that textures are always centered in the middle of the canvas.
            texture_color_mask, texture_alpha_mask = self._scale_and_translate.transform(
                texture,
                scale=radius / self._texture_radius,
                translate=torch.stack([x - self.canvas_size / 2, y - self.canvas_size / 2], dim=-1),
                center=self._canvas_center,
                mode=mode,
                padding_mode="zeros",
            ).chunk(2, dim=1)

            foreground_alpha_mask = texture_alpha_mask * inv_fade + (1 - texture_alpha_mask) * foreground_alpha_mask
            foreground_color_mask = texture_color_mask * inv_fade + (1 - texture_alpha_mask) * foreground_color_mask

        foreground = foreground_color_mask * base_color[..., None, None]
        alpha_mask = foreground_alpha_mask * self.alpha_mask_scale

        if self.apply_closing:
            foreground = kornia.morphology.dilation(foreground, self._morph_kernel)
            alpha_mask = kornia.morphology.erosion(alpha_mask, self._morph_kernel)

        foreground = kornia.filters.box_blur(foreground, 2)

        return foreground, alpha_mask

    def _create_regular_polygon(self, radius: Tensor | float, sides: int) -> Tensor:
        """Creates regular polygons with given radius that are centered at the origin (0, 0).

        Args:
            radius (Tensor or float): Radius of the polygons, if Tensor shape should be [B].
            sides (int): Number of the polygon sides.

        Returns:
            Tensor: Vertices of regular polygons of shape [B, sides, 2].
        """
        if not isinstance(radius, Tensor):
            radius = torch.tensor([radius], device=self.device)

        # Since end is inclusive we're using steps=sides + 1 and discarding the last element,
        # which is the same as the first angle zero (0 rad == 2*pi rad).
        angles = torch.linspace(start=0, end=2 * torch.pi, steps=sides + 1, device=self.device)[:-1]
        angles_cos = angles.cos()
        angles_sin = angles.sin()

        vertices = torch.stack([radius.outer(angles_cos), radius.outer(angles_sin)], dim=-1)
        return vertices

    def _deform_polygon(self, vertices: Tensor, variance: float, depth: int) -> Tensor:
        """Recursively deforms a polygon by adding new points with random displacements to the vertices.

        New vertices are initially placed in the middle of the edges and then moved randomly based on the variance.

        Args:
            vertices (Tensor): Input vertices of shape [B, V, 2], where B is batch size and V is vertex count.
            variance (float): Variance used for random displacements.
            depth (int): Recursion depth, determining the number of times to subdivide edges.

        Returns:
            Tensor: Deformed vertices of shape [B, V * 2^depth, 2].
        """
        if depth == 0:
            return vertices

        batch_size = vertices.shape[0]

        a = vertices
        c = vertices.roll(shifts=-1, dims=1)

        displacement = self.random.normal(0.0, torch.full_like(vertices, variance), size=vertices.shape, device=self.device)
        displacement = displacement.clip(-2 * variance, 2 * variance)

        b = (a + c) / 2 + displacement

        new_vertices = torch.stack([a, b], dim=2).reshape(batch_size, -1, 2)

        return self._deform_polygon(new_vertices, variance * self.variance_decay, depth - 1)

    def _denormalize(self, brush_params: Tensor) -> Tensor:
        """Convert normalized brush parameters to pixel coordinates and color values.

        Args:
            brush_params: Normalized parameters [B, 10] with values in [0, 1].

        Returns:
            Denormalized parameters [B, 10] with positions in pixels, radius in pixels, and colors in [0, 255].
        """
        points = brush_params[:, :6]
        points = points * (self.canvas_size - 1)

        radius = brush_params[:, 6:7]
        radius = self.min_radius + radius * (self.max_radius - self.min_radius)

        # Minimal brush stroke radius must be at least of size 1.
        radius = radius.clamp(min=1)

        color = brush_params[:, 7:10]
        color = color * 255

        return torch.cat([points, radius, color], dim=1)


class TextureBrush(Brush):
    """
    Texture-based brush that renders colored, gradient strokes on a black canvas.

    The brush stroke is rendered using a texture selected based on stroke size and orientation,
    and is colored with a smooth gradient between two RGB colors (from start to the end).
    """

    def __init__(
        self,
        canvas_size: int,
        large_vertical: torch.Tensor,
        large_horizontal: torch.Tensor,
        small_vertical: torch.Tensor,
        small_horizontal: torch.Tensor,
        *,
        min_size_fraction: float = 0.01,
        max_size_fraction: float = 1.0,
        apply_closing: bool = False,
        device: str = "cpu",
    ):
        """Validate params and create TextureBrush.

        IMPORTANT NOTE!!! min_size_fraction must be high enough so that strokes with width
        and height params set to 0 are still visible.

        Args:
            canvas_size (int): Size of the square output canvas (canvas_size x canvas_size).
            large_vertical (Tensor): Normalized grayscale texture [1, H, W] for large vertical strokes.
            large_horizontal (Tensor): Normalized grayscale texture [1, H, W] for large horizontal strokes.
            small_vertical (Tensor): Normalized grayscale texture [1, H, W] for small vertical strokes.
            small_horizontal (Tensor): Normalized grayscale texture [1, H, W] for small horizontal strokes.
            min_size_fraction (float, optional): Minimum stroke size as fraction of canvas. Default 0.01.
            max_size_fraction (float, optional): Maximum stroke size as fraction of canvas. Default 1.0.
            apply_closing (bool, optional): If True, apply a small morphological closing step. Default False.
            device (str, optional): Device for internal tensors, e.g. "cpu" or "cuda". Default "cpu".
        """
        if max_size_fraction < min_size_fraction:
            raise ValueError(
                f"Max size must be bigger than the min size. "
                f"Got: max = {max_size_fraction:.4f}, min = {min_size_fraction:.4f}"
            )

        for t in [large_vertical, large_horizontal, small_vertical, small_horizontal]:
            if t.shape[0] != 1 or len(t.shape) != 3 or t.min() < 0 or t.max() > 1:
                raise ValueError("Textures must be normalized grayscale image with only one channel dimension.")

        self._canvas_size = canvas_size
        self._canvas_center = torch.tensor([canvas_size / 2, canvas_size / 2], dtype=torch.float32, device=device)

        self.min_size = canvas_size * min_size_fraction
        self.max_size = canvas_size * max_size_fraction

        self.kernel = torch.ones(3, 3, dtype=torch.float32, device=device)
        self.apply_closing = apply_closing
        self._morph_kernel = torch.ones(2, 2, dtype=torch.float32, device=device)

        self.textures = {
            "large_vertical": large_vertical,
            "large_horizontal": large_horizontal,
            "small_vertical": small_vertical,
            "small_horizontal": small_horizontal,
        }
        for n, t in self.textures.items():
            # This is done for smoother textures without holes and for better gradient propagation.
            self.textures[n] = kornia.morphology.dilation(t.to(device).unsqueeze(0), self.kernel).squeeze(0)

        self.device = device

    @property
    def brush_params_count(self) -> int:
        return 11

    @property
    def canvas_size(self) -> int:
        return self._canvas_size

    def render_brush_stroke(self, brush_params: Tensor, **kwargs) -> tuple[Tensor, Tensor]:
        """Renders normalized texture brush strokes on a black canvas.

        Args:
            brush_params (Tensor): Batched normalized brush parameters of shape [B, 11] in a range of [0, 1].
                Order of the params is [xc, yc, w, h, angle, R0, G0, B0, R1, G1, B1]

        Returns:
            tuple[Tensor, Tensor]: A tuple containing:
                - Foreground: Brush stroke rendered on black canvas of shape [B, 3, canvas_size, canvas_size].
                - Alpha mask: Alpha mask of shape [B, 1, canvas_size, canvas_size].
        """
        if brush_params.shape[1] != self.brush_params_count:
            raise ValueError(f"Invalid shape of brush parameters. Expected [B, 11] but found {brush_params.shape}")

        batch_size = brush_params.shape[0]

        denormalized_params = self._denormalize(brush_params)

        xc, yc, w, h, angle = denormalized_params[:, :5].T
        color_start = denormalized_params[:, 5:8]
        color_end = denormalized_params[:, 8:11]

        # Get raw textures and apply color gradient for each batch element.
        h_list = h.tolist()
        w_list = w.tolist()
        colored_textures = []
        for i in range(batch_size):
            colored_texture = self._get_colored_texture(h_list[i], w_list[i], color_start[i], color_end[i])
            colored_textures.append(colored_texture)

        # Transform colored textures to final positions/orientations and scale.
        transformed = self._transform_textures(denormalized_params, colored_textures)
        foreground, alpha_mask = transformed.split([3, 1], dim=1)

        # Set black background where alpha is 0, this is done to be consistent with other brush strokes.
        foreground = torch.where(
            alpha_mask > 0,
            foreground,
            torch.full_like(foreground, 0.0)
        )

        if self.apply_closing:
            foreground = kornia.morphology.dilation(foreground, self._morph_kernel)
            alpha_mask = kornia.morphology.erosion(alpha_mask, self._morph_kernel)

        return foreground, alpha_mask

    def _get_colored_texture(
        self,
        stroke_height: float,
        stroke_width: float,
        color_start: Tensor,
        color_end: Tensor,
    ) -> Tensor:
        """Select appropriate texture and apply color gradient.

        Args:
            stroke_height (float): Stroke height in pixels.
            stroke_width (float): Stroke width in pixels.
            color_start (Tensor): Denormalized RGB start color of shape [3] with values in [0, 255].
            color_end (Tensor): Denormalized RGB end color of shape [3] with values in [0, 255].

        Returns:
            Tensor: Colored RGBA texture of shape [4, H, W].
        """
        is_small = stroke_height * stroke_width / (self.canvas_size ** 2) <= 0.1
        is_vertical = stroke_height > stroke_width

        if is_small:
            texture = (
                self.textures["small_vertical"]
                if is_vertical
                else self.textures["small_horizontal"]
            )
        else:
            texture = (
                self.textures["large_vertical"]
                if is_vertical
                else self.textures["large_horizontal"]
            )

        # Compute color gradient along stroke direction
        texture_height, texture_width = texture.shape[-2:]
        if is_vertical:
            gradient = torch.linspace(0, 1, texture_height, device=self.device).view(1, texture_height, 1).expand(1, texture_height, texture_width)
        else:
            gradient = torch.linspace(0, 1, texture_width, device=self.device).view(1, 1, texture_width).expand(1, texture_height, texture_width)
        gradient_color = color_start.view(3, 1, 1) + gradient * (color_end - color_start).view(3, 1, 1)

        foreground = texture * gradient_color
        alpha = texture

        return torch.cat([foreground, alpha], dim=0)

    def _transform_textures(self, brush_params: torch.Tensor, colored_textures: list[torch.Tensor]) -> torch.Tensor:
        """Transform colored RGBA textures to final positions, orientations, and sizes.

        Args:
            brush_params (Tensor): Denormalized brush parameters of shape [B, 11].
            colored_textures (List[Tensor]): List of B colored RGBA textures, each of shape [4, H, W].

        Returns:
            Tensor: Transformed textures of shape [B, 4, canvas_size, canvas_size].
        """
        batch_size = brush_params.shape[0]
        xc, yc, w, h, angle = brush_params.T[:5]

        scale = torch.stack([
            w / torch.tensor([t.shape[-1] for t in colored_textures], device=w.device),
            h / torch.tensor([t.shape[-2] for t in colored_textures], device=h.device),
        ], dim=1)
        texture_center = torch.stack([
            torch.tensor([t.shape[-1] / 2 for t in colored_textures], device=w.device),
            torch.tensor([t.shape[-2] / 2 for t in colored_textures], device=h.device),
        ], dim=1)
        # Move to the target xc and yc from the texture center
        target_pos = torch.stack([xc, yc], dim=1)
        translation = target_pos - texture_center

        M = kornia.geometry.get_affine_matrix2d(
            translations=translation,
            center=texture_center,
            scale=scale,
            angle=angle,
        )[:, :2, :]

        # Apply transformations (we need to transform each texture separately due to different shapes)
        transformed_textures = []
        for i in range(batch_size):
            curr_texture = kornia.geometry.warp_affine(
                colored_textures[i][None],
                M[i:i + 1],
                dsize=(self.canvas_size, self.canvas_size),
                mode="bilinear",
                padding_mode="zeros",
                # This is important for more smooth textures.
                align_corners=False,
            )
            transformed_textures.append(curr_texture)
        transformed_textures = torch.cat(transformed_textures)

        return transformed_textures

    def _denormalize(self, brush_params: Tensor) -> Tensor:
        """Convert normalized brush parameters to pixel coordinates and color values.

        Args:
            brush_params: Normalized parameters [B, 11] with values in [0, 1].

        Returns:
            Denormalized parameters [B, 11] with center/size in pixels, angle in degrees, and colors in [0, 255].
        """
        normalized_center = brush_params[:, :2]
        center = normalized_center * (self.canvas_size - 1)

        normalized_size = brush_params[:, 2:4]
        size = self.min_size + normalized_size * (self.max_size - self.min_size)

        normalized_angle = brush_params[:, 4:5]
        angle = 180 * normalized_angle

        normalized_color = brush_params[:, 5:]
        color = normalized_color * 255

        return torch.cat([center, size, angle, color], dim=1)


class RectangleBrush(Brush):
    """Brush that renders axis-aligned rectangles with grainy watercolor effects.

    Uses a 3-texture system applied sequentially:
    1. Coarse grain - High-scale blurred noise for paper texture
    2. Mottling - Medium-scale variation for pigment pooling effects
    3. Fine grain - Pixel-level noise for subtle texture detail
    """

    # Pre-generated textures for grainy watercolor effect (high-scale to fine grain).
    # Each element has shape [N, 1, S, S].
    _textures: list[Tensor]

    def __init__(
        self,
        canvas_size: int,
        *,
        n_textures: int = 20,
        texture_intensities: list[tuple[float, float]] | None = None,
        color_desaturation: float = 0.2,
        min_size_fraction: float = 0.01,
        max_size_fraction: float = 1.0,
        boundary_width: float = 1.0,
        boundary_darkness: float = 0.6,
        max_size_factor: float = 2.0,
        brightness_scale: float = 0.5,
        device: str = "cpu",
        seed: int | None = 1,
    ):
        """Initialize a rectangle brush with grainy watercolor effects.

        The brush uses a 3-texture system stored in textures list that are applied in this order:
        1. Coarse grain (high-scale blurred noise)
        2. Mottling (pigment pooling effect)
        3. Fine grain (pixel-level variation)

        Each texture type has `n_textures` variants for randomization.

        Args:
            canvas_size (int): Size of the square output canvas.
            n_textures (int): Number of pre-generated texture variants per type.
            texture_intensities (list | None): List of (lighten, darken) intensity
                tuples for each texture layer [coarse grain, mottling, fine grain].
                Defaults to [(0.5, 0.35), (0.15, 0.15), (0.5, 0.5)].
            color_desaturation (float): How much to mute colors towards grayscale (0-1).
                0 = vivid colors, 1 = fully grayscale.
            min_size_fraction (float): Minimum rectangle size as fraction of canvas.
            max_size_fraction (float): Maximum rectangle size as fraction of canvas.
            boundary_width (float): Width of the dark boundary in pixels.
            boundary_darkness (float): How much to darken the boundary (0-1). Higher values
                produce darker boundaries.
            max_size_factor (float): Controls how much additional texture is applied to smaller
                strokes. It ensures that small strokes display a visible and meaningful
                texture pattern.
            brightness_scale (float): Limits the amount of darkening applied to the bright
                strokes to preserve their brightness and vice versa with dark strokes.
            device (str): Device for PyTorch tensors. Default: "cpu".
            seed (int | None): Seed for random generator. Default: 1.
        """
        if max_size_fraction < min_size_fraction:
            raise ValueError(
                f"Max size must be bigger than the min size. "
                f"Got: max = {max_size_fraction:.4f}, min = {min_size_fraction:.4f}"
            )

        if texture_intensities is None:
            texture_intensities = [(0.5, 0.35), (0.15, 0.15), (0.5, 0.5)]

        if len(texture_intensities) != 3:
            raise ValueError(f"Texture intesities must have exactly 3 values")

        self._canvas_size = canvas_size
        self.n_textures = n_textures
        self.max_size_factor = max_size_factor
        self.brightness_scale = brightness_scale
        self.texture_intensities = texture_intensities
        self.color_desaturation = color_desaturation

        self.min_size = max(1.0, min_size_fraction * canvas_size)
        self.max_size = max(1.0, max_size_fraction * canvas_size)

        self.device = device
        self.seed = seed
        self.random = RandomWrapper(seed=seed)

        self.boundary_width = boundary_width
        self.boundary_darkness = boundary_darkness

        self._initialize_grain_textures()

    def _initialize_grain_textures(self):
        """Pre-generate coarse grain, mottling, and fine grain textures."""
        coarse_textures = []
        mottling_textures = []
        fine_grain_textures = []

        for _ in range(self.n_textures):
            coarse_textures.append(self._generate_coarse_texture())
            mottling_textures.append(self._generate_mottling_texture())
            fine_grain_textures.append(self._generate_fine_grain_texture())

        self._textures = [
            torch.stack(coarse_textures, dim=0),
            torch.stack(mottling_textures, dim=0),
            torch.stack(fine_grain_textures, dim=0),
        ]

    def _generate_coarse_texture(self) -> Tensor:
        return self._generate_multiscale_noise(
            scales=(2, 4, 6, 10, 16),
            weights=(0.15, 0.22, 0.28, 0.22, 0.13),
            output_min=0.3,
            output_range=0.4,
        )

    def _generate_mottling_texture(self) -> Tensor:
        return self._generate_multiscale_noise(
            scales=(32, 64),
            weights=(0.6, 0.4),
            output_min=0.4,
            output_range=0.2,
        )

    def _generate_fine_grain_texture(self) -> Tensor:
        return self._generate_multiscale_noise(
            scales=(1, 2, 3),
            weights=(0.5, 0.3, 0.2),
            output_min=0.425,
            output_range=0.15,
        )

    def _generate_multiscale_noise(
        self,
        scales: Sequence[int],
        weights: Sequence[float],
        output_min: float,
        output_range: float,
    ) -> Tensor:
        """Generate multi-scale noise texture normalized to [output_min, output_min + output_range].

        Args:
            scales (Sequence[int]): Noise resolutions to generate and blend.
            weights (Sequence[float]): Blending weight for each scale.
            output_min (float): Minimum value of the output range.
            output_range (float): Size of the output range.

        Returns:
            Tensor of shape [1, canvas_size, canvas_size].
        """
        texture = torch.zeros(1, self.canvas_size, self.canvas_size, device=self.device)

        for scale, weight in zip(scales, weights):
            if scale == 1:
                noise = self.random.rand(size=(1, 1, self.canvas_size, self.canvas_size), device=self.device)
                texture += noise.squeeze(0) * weight
            else:
                small_noise = self.random.rand(size=(1, 1, scale, scale), device=self.device)
                upscaled = torch.nn.functional.interpolate(
                    small_noise, size=self.canvas_size, mode='bilinear', align_corners=False
                )
                texture += upscaled.squeeze(0) * weight

        texture_min = texture.min()
        texture_max = texture.max()
        texture = (texture - texture_min) / (texture_max - texture_min + 1e-9)
        return output_min + texture * output_range

    @property
    def brush_params_count(self) -> int:
        return 7

    @property
    def canvas_size(self) -> int:
        return self._canvas_size

    def render_brush_stroke(self, brush_params: Tensor, **kwargs) -> tuple[Tensor, Tensor]:
        """Renders axis-aligned rectangles with grainy watercolor effects.

        Args:
            brush_params: Batched normalized brush parameters [B, 7] in range [0, 1].
                Order: [xc, yc, width, height, r, g, b]

        Returns:
            Foreground [B, 3, S, S] in [0, 255] and alpha mask [B, 1, S, S] in [0, 1].
        """
        if brush_params.shape[1] != self.brush_params_count:
            raise ValueError(f"Invalid shape of brush parameters. Expected [B, 7] but found {brush_params.shape}")

        batch_size = brush_params.shape[0]

        denormalized_params = self._denormalize(brush_params)
        xc, yc, w, h = denormalized_params[:, :4].T
        colors = denormalized_params[:, 4:7]

        alpha_mask, boundary_mask = self._create_rectangle_masks(xc, yc, w, h, batch_size)

        colors = self._desaturate_colors(colors)

        # Compute factors based on size and brightness.
        size_factor = self._compute_size_factor(w, h, batch_size)
        darken_scale, lighten_scale = self._compute_brightness_scales(colors)

        # Apply grainy effects on the rectangle.
        texture_indices = self.random.rand_int(0, self.n_textures, (batch_size,), device=self.device)
        foreground = self._apply_texture_effects(
            colors=colors,
            size_factor=size_factor,
            darken_scale=darken_scale,
            lighten_scale=lighten_scale,
            texture_indices=texture_indices,
            boundary_mask=boundary_mask,
        )
        foreground = (foreground * alpha_mask).clamp(0, 255)

        return foreground, alpha_mask

    def _apply_texture_effects(
        self,
        colors: Tensor,
        size_factor: Tensor,
        darken_scale: Tensor,
        lighten_scale: Tensor,
        texture_indices: Tensor,
        boundary_mask: Tensor,
    ) -> Tensor:
        """Apply grain, mottling, fine grain, and boundary effects to base color.

        Args:
            colors (Tensor): Base colors [B, 3] in [0, 255].
            size_factor (Tensor): Grain intensity multiplier [B, 1, 1, 1].
            darken_scale (Tensor): Scaling for darkening effects [B, 1, 1, 1].
            lighten_scale (Tensor): Scaling for lightening effects [B, 1, 1, 1].
            texture_indices (Tensor): Indices into pre-generated textures [B].
            boundary_mask (Tensor): Mask for boundary darkening [B, 1, S, S].

        Returns:
            Textured color [B, 3, S, S].
        """
        result = colors[:, :, None, None].expand(-1, -1, self.canvas_size, self.canvas_size)

        # Apply all textures in sequence (from high-scale grain to fine grain)
        for textures, (lighten_intensity, darken_intensity) in zip(self._textures, self.texture_intensities):
            texture = textures[texture_indices]
            result = self._apply_bidirectional_texture(
                result, texture, lighten_intensity, darken_intensity,
                size_factor, darken_scale, lighten_scale
            )

        # Apply boundary darkening using the last (finest grain) texture
        last_texture = self._textures[-1][texture_indices]
        boundary_darken = self.boundary_darkness * last_texture
        result = result * (1 - boundary_darken * boundary_mask)

        return result

    def _apply_bidirectional_texture(
        self,
        color: Tensor,
        texture: Tensor,
        lighten_intensity: float,
        darken_intensity: float,
        size_factor: Tensor,
        darken_scale: Tensor,
        lighten_scale: Tensor,
    ) -> Tensor:
        lighten_amount = (texture - 0.5).clamp(min=0) * 2 * lighten_intensity * size_factor * lighten_scale
        darken_amount = (0.5 - texture).clamp(min=0) * 2 * darken_intensity * size_factor * darken_scale
        color = color + (255 - color) * lighten_amount
        color = color * (1 - darken_amount)
        return color

    def _create_rectangle_masks(
        self, xc: Tensor, yc: Tensor, w: Tensor, h: Tensor, batch_size: int
    ) -> tuple[Tensor, Tensor]:
        """Create alpha mask and boundary mask for the rectangles.

        Returns:
            alpha_mask: [B, 1, S, S] with 1 inside rectangle (including edges), 0 outside.
            boundary_mask: [B, 1, S, S] with 1 at boundary, 0 elsewhere.
        """
        y_coords = torch.arange(self.canvas_size, device=self.device, dtype=torch.float32)
        x_coords = torch.arange(self.canvas_size, device=self.device, dtype=torch.float32)
        yy, xx = torch.meshgrid(y_coords, x_coords, indexing="ij")
        xx = xx.unsqueeze(0)
        yy = yy.unsqueeze(0)

        x_min = (xc - w / 2).view(batch_size, 1, 1)
        x_max = (xc + w / 2).view(batch_size, 1, 1)
        y_min = (yc - h / 2).view(batch_size, 1, 1)
        y_max = (yc + h / 2).view(batch_size, 1, 1)

        inside_x = (xx >= x_min) & (xx < x_max)
        inside_y = (yy >= y_min) & (yy < y_max)
        alpha_mask = (inside_x & inside_y).float().unsqueeze(1)

        inside_x_inner = (xx >= x_min + self.boundary_width) & (xx < x_max - self.boundary_width)
        inside_y_inner = (yy >= y_min + self.boundary_width) & (yy < y_max - self.boundary_width)
        inner_mask = (inside_x_inner & inside_y_inner).float().unsqueeze(1)
        boundary_mask = alpha_mask - inner_mask

        return alpha_mask, boundary_mask

    def _desaturate_colors(self, colors: Tensor) -> Tensor:
        gray = colors.mean(dim=1, keepdim=True)
        return colors + (gray - colors) * self.color_desaturation

    def _compute_size_factor(self, w: Tensor, h: Tensor, batch_size: int) -> Tensor:
        """Calculate grain intensity multiplier based on rectangle size.

        Smaller rectangles get higher grain intensity to maintain texture visibility.
        """
        normalized_area = (w / self.canvas_size) * (h / self.canvas_size)
        size_factor = (1.0 / (normalized_area.sqrt() + 1e-7)).clamp(max=self.max_size_factor)
        return size_factor.view(batch_size, 1, 1, 1)

    def _compute_brightness_scales(self, colors: Tensor) -> tuple[Tensor, Tensor]:
        """Calculate brightness-adaptive scaling factors for texture effects.

        Light colors get reduced darkening, dark colors get reduced lightening
        to avoid harsh contrast.

        Returns:
            darken_scale: [B, 1, 1, 1] scaling factor for darkening effects.
            lighten_scale: [B, 1, 1, 1] scaling factor for lightening effects.
        """
        brightness = colors.mean(dim=1, keepdim=True) / 255.0
        darken_scale = 1.0 - brightness.sqrt() * self.brightness_scale
        lighten_scale = 1.0 - self.brightness_scale + brightness.sqrt() * self.brightness_scale
        return darken_scale[:, :, None, None], lighten_scale[:, :, None, None]

    def _denormalize(self, brush_params: Tensor) -> Tensor:
        """Convert normalized [0, 1] parameters to pixel coordinates and [0, 255] colors."""
        center = brush_params[:, :2] * (self.canvas_size - 1)

        size = self.min_size + brush_params[:, 2:4] * (self.max_size - self.min_size)

        color = brush_params[:, 4:7] * 255

        return torch.cat([center, size, color], dim=1)


class DifferentiableBrush(nn.Module, Brush, ABC):
    """
    Parent class for the trainable brushes whose render_brush_stroke method is differentiable with respect to the brush_params.

    Note that differentiable brush parameters may produce output outside [0, 255] range. We're on purpose not clamping
    the final tensor to not cut away gradient.
    """


class GeneralNeuralRenderer(DifferentiableBrush):
    """General-purpose differentiable brush that uses a neural network to render RGBA images."""

    def __init__(
        self,
        brush_params_count: int,
        canvas_size: int = 128,
        nn_renderer: nn.Module | None = None,
        device: str = "cpu",
    ):
        """Initialize the GeneralNeuralRenderer.

        Args:
            brush_params_count (int): Number of input brush parameters. This defines
                the input dimension of the neural network.
            canvas_size (int): Size of the square output canvas. Supported values are
                32 (uses ModulatedPixelShuffleNetLight) and 128 (uses ModulatedPixelShuffleNet).
                Default: 128.
            nn_renderer (nn.Module | None): Custom neural network for rendering. If None,
                a default ModulatedPixelShuffleNet (for canvas_size=128) or
                ModulatedPixelShuffleNetLight (for canvas_size=32) is created. The network
                should accept input of shape [B, brush_params_count] and output shape
                [B, 4, canvas_size, canvas_size]. Default: None.
            device (str): Device for PyTorch tensors ('cpu' or 'cuda'). Default: 'cpu'.
        """
        super().__init__()

        if canvas_size <= 0:
            raise ValueError(f"canvas_size must be positive, got {canvas_size}.")

        self._brush_params_count = int(brush_params_count)
        self._canvas_size = int(canvas_size)
        self.device = device

        if nn_renderer is None:
            if canvas_size == 128:
                nn_renderer = ModulatedPixelShuffleNet(
                    input_dim=self._brush_params_count,
                    out_channels=4,
                    device=device,
                )
            elif canvas_size == 32:
                nn_renderer = ModulatedPixelShuffleNetLight(
                    input_dim=self._brush_params_count,
                    out_channels=4,
                    device=device,
                )
            else:
                raise ValueError(f"Unsupported canvas_size: {canvas_size}. Supported: 32, 128")

        self.nn_renderer = nn_renderer

    @property
    def canvas_size(self) -> int:
        return self._canvas_size

    @property
    def brush_params_count(self) -> int:
        return self._brush_params_count

    def render_brush_stroke(self, brush_params: Tensor, **kwargs) -> tuple[Tensor, Tensor]:
        rgba_img = self.nn_renderer(brush_params)  # [B, 4, S, S]

        assert self.canvas_size == rgba_img.shape[-1] and self.canvas_size == rgba_img.shape[-2]

        foreground, alpha_mask = rgba_img.split([3, 1], dim=1)
        foreground = foreground * 255

        return foreground, alpha_mask


class ColorMaskNeuralRenderer(DifferentiableBrush):
    """
    Differentiable brush that can be trained to imitate brushes that use "color mask" and alpha mask to render final image.

    Color mask is of the same shape as canvas with values in range [0, 1] which represents spectrum between black
    (when mask is equal to 0) and brush stroke's color (when mask is equal to 1).

    The underlying neural network model predicts this color mask from brush stroke parameters without color ones.
    This color mask is then combined with brush's color to compute final rendered image.

    The motivation for this limitation, rather than simply learning to predict rendered brush stroke,
    is simpler and faster training for the brushes like WatercolorBrush that don't do any other color mixing except
    for the black and brush stroke's color (the color mask has only one channel instead of rendered brush stroke that has 3).
    """
    def __init__(
        self,
        brush_params_count: int,
        color_params_indices: list[int],
        canvas_size: int = 128,
        rasterization_network: nn.Module | None = None,
        device: str = "cpu"
    ):
        """Initialize the ColorMaskNeuralRenderer.

        Args:
            brush_params_count (int): Total number of brush parameters including color.
                The network input will exclude color parameters (uses brush_params_count - 3).
            color_params_indices (list[int]): List of exactly 3 indices specifying the
                positions of R, G, B color parameters (in that order) within the brush
                parameters tensor. These parameters are excluded from network input and
                used directly to color the predicted mask.
            canvas_size (int): Size of the square output canvas. Supported values are
                32 (uses ModulatedPixelShuffleNetLight) and 128 (uses ModulatedPixelShuffleNet).
                Default: 128.
            rasterization_network (nn.Module | None): Custom neural network for predicting
                color mask and alpha mask. If None, a default ModulatedPixelShuffleNet (for
                canvas_size=128) or ModulatedPixelShuffleNetLight (for canvas_size=32) is created.
                The network should accept input of shape [B, brush_params_count - 3] and
                output shape [B, 2, canvas_size, canvas_size] where channels are
                [color_mask, alpha_mask]. Default: None.
            device (str): Device for PyTorch tensors ('cpu' or 'cuda'). Default: 'cpu'.
        """
        super().__init__()

        if canvas_size <= 0:
            raise ValueError(f"canvas_size must be positive, got {canvas_size}.")

        if len(color_params_indices) != 3:
            raise ValueError("RGB color param indices must be of size 3")

        self._brush_params_count = brush_params_count
        self._canvas_size = canvas_size
        self.color_params_indices = color_params_indices
        self.device = device

        self.raster_nn_input_indices = [i for i in range(brush_params_count) if i not in color_params_indices]

        if rasterization_network is None:
            if canvas_size == 128:
                rasterization_network = ModulatedPixelShuffleNet(
                    len(self.raster_nn_input_indices),
                    out_channels=2,
                    device=device,
                )
            elif canvas_size == 32:
                rasterization_network = ModulatedPixelShuffleNetLight(
                    len(self.raster_nn_input_indices),
                    out_channels=2,
                    device=device,
                )
            else:
                raise ValueError(f"Unsupported canvas_size: {canvas_size}. Supported: 32, 128")

        self.rasterization_network = rasterization_network

    @property
    def canvas_size(self):
        return self._canvas_size

    @property
    def brush_params_count(self) -> int:
        return self._brush_params_count

    def render_brush_stroke(self, brush_params: Tensor, **kwargs) -> tuple[Tensor, Tensor]:
        raster_input = brush_params[:, self.raster_nn_input_indices]

        # Each mask has shape of [B, 1, canvas_size, canvas_size].
        color_mask, alpha_mask = self.rasterization_network(raster_input).chunk(2, dim=1)

        assert self.canvas_size == color_mask.shape[-1] and self.canvas_size == color_mask.shape[-2]

        color = brush_params[:, self.color_params_indices] * 255  # [B, 3]
        foreground = color_mask * color[..., None, None]

        return foreground, alpha_mask
