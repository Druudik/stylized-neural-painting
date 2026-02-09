import kornia
import torch
import torch.nn.functional as F
from torch import Tensor


class ScaleAndTranslate:
    """
    Optimized scale and translate transformation.
    """

    def __init__(self, canvas_size: int, device: str):
        self.canvas_size = canvas_size
        self.device = device

        y, x = torch.meshgrid(
            [torch.linspace(-1, 1, canvas_size, device=device),
             torch.linspace(-1, 1, canvas_size, device=device)],
            indexing="ij",
        )
        ones = torch.ones_like(x)
        self._base_grid = torch.stack([x, y, ones], dim=2).unsqueeze(0)  # [1, S, S, 3]
        self._base_grid = self._base_grid.view(1, canvas_size * canvas_size, 3)  # [1, S * S, 3]

    def transform(
        self,
        image: Tensor,
        scale: Tensor,
        translate: Tensor,
        center: Tensor,
        mode: str,
        padding_mode: str,
        align_corners: bool = True
    ) -> Tensor:
        """Performs optimized scale and translate transformation of the input image.

        This function takes advantage of the fact that we're only scaling images by the same scale in all directions
        and translating images without any rotations. This allows for faster computation of the matrix inverse operations
        and other optimizations.

        Args:
            image (Tensor): Squared images of shape [B, C, S, S].
            scale (Tensor): Scale by which the image will be increased in all directions of shape [B].
            translate (Tensor): Vector by which image will be translated of shape [B, 2].
            center (Tensor): Center of the canvas of shape [2] or [B, 2].
            mode (str): Interpolation mode.
                For more details check out docs of torch.nn.functional.grid_sample
            padding_mode (str): Padding mode to be used for outside grid values. Default: 'zeros'.
                For more details check out docs of torch.nn.functional.grid_sample
            align_corners (bool): If True, consider -1 and 1 to refer to the centers of the corner pixels rather
                than the image corners. For more details check out docs of torch.nn.functional.grid_sample
                Default: True.

        Returns:
            Transformed image of the same shape as the input [B, C, S, S].
        """
        if len(image.shape) != 4:
            raise ValueError("Invalid image dimensions")

        if not (self.canvas_size == image.shape[-1] == image.shape[-2]):
            raise ValueError(f"Only square images of size {self.canvas_size} are supported.")

        canvas_size = image.shape[-1]
        batch_size = scale.shape[0]

        if len(center.shape) == 1:
            center = center.expand(batch_size, -1)

        affine_matrix = torch.zeros(batch_size, 3, 3, device=self.device, dtype=torch.float32)
        affine_matrix[:, 0, 0] = scale
        affine_matrix[:, 1, 1] = scale
        affine_matrix[:, 2, 2] = 1

        # Calculate inverse translation part so that center will stay centered after scaling and rotating.
        center_inv = -(affine_matrix[:, :2, :2] @ center.unsqueeze(-1)).squeeze(-1)
        affine_matrix[:, :2, 2] += center + translate + center_inv

        # Normalize affine matrix since it's expected by the grid_sample function.
        norm_scale = 2 / (canvas_size - 1)
        affine_matrix[:, :2, 2] = affine_matrix[:, :2, 2] * norm_scale + affine_matrix[:, 0, 0].unsqueeze(1) - 1

        # Inverse affine matrix.
        inverse_scale = 1 / affine_matrix[:, 0, 0]
        affine_matrix[:, 0, 0] = inverse_scale
        affine_matrix[:, 1, 1] = inverse_scale
        affine_matrix[:, :2, 2] = -affine_matrix[:, :2, 2] * inverse_scale.unsqueeze(1)

        # Compute affine grid.
        affine_matrix = affine_matrix[:, :2, :]
        affine_grid = torch.matmul(self._base_grid, affine_matrix.transpose(1, 2))
        affine_grid = affine_grid.view(-1, canvas_size, canvas_size, 2)

        return F.grid_sample(
            image,
            affine_grid,
            mode=mode,
            padding_mode=padding_mode,
            align_corners=align_corners
        )


def apply_closing_to_brush_stroke(
    foreground: Tensor,
    alpha_mask: Tensor,
    kernel: Tensor,
) -> tuple[Tensor, Tensor]:
    """Apply closing transform to brush stroke: dilation on foreground, erosion on alpha mask.

    Args:
        foreground (Tensor): Brush stroke foreground of shape [B, 3, S, S] in range [0, 255].
        alpha_mask (Tensor): Brush stroke alpha mask of shape [B, 1, S, S] in range [0, 1].
        kernel (Tensor): Morphological kernel of shape [K, K].

    Returns:
        Tuple of (transformed_foreground, transformed_alpha_mask) with same shapes and ranges.
    """
    transformed_foreground = kornia.morphology.dilation(foreground, kernel)
    transformed_alpha_mask = kornia.morphology.erosion(alpha_mask, kernel)
    return transformed_foreground, transformed_alpha_mask


def create_affine_matrix(scale: Tensor, angle: Tensor, translate: Tensor, center: Tensor) -> Tensor:
    """
    Creates combined affine matrix that is equivalent to applying scale operation first (while keeping the center centered),
    then rotate and finally translate.

    Args:
        scale (Tensor): Scale by which the image will be increased in all directions of shape [B].
        angle (Tensor): Angle in radians by which the image will be rotated of shape [B].
            Positive values mean counter-clockwise rotation.
        translate (Tensor): Vector by which image will be translated of shape [B, 2].
        center (Tensor): Center of the canvas of shape [2] or [B, 2].

    Returns:
        Tensor holding affine matrices of shape [B, 2, 3].
    """
    batch_size = scale.shape[0]
    device = scale.device

    if len(center.shape) == 1:
        center = center.expand(batch_size, -1)

    cos_theta = torch.cos(angle)
    sin_theta = torch.sin(angle)

    affine_matrix = torch.zeros(batch_size, 2, 3, device=device, dtype=torch.float32)
    affine_matrix[:, 0, 0] = scale * cos_theta
    affine_matrix[:, 0, 1] = -scale * sin_theta
    affine_matrix[:, 1, 0] = scale * sin_theta
    affine_matrix[:, 1, 1] = scale * cos_theta

    # Calculate inverse translation part so that center will stay centered after scaling and rotating.
    center_inv = -(affine_matrix[:, :2, :2] @ center.unsqueeze(-1)).squeeze(-1)
    affine_matrix[:, :2, 2] += center + translate + center_inv

    return affine_matrix


def affine(image: Tensor, matrix: Tensor, mode: str, padding_mode: str, align_corners: bool = True) -> Tensor:
    """Performs affine transformation of the input image.

    Args:
        image (Tensor): Squared images of shape [B, C, S, S].
        matrix (Tensor): Affine matrix of shape [B, 2, 3].
        mode (str): Interpolation mode.
            For more details check out docs of torch.nn.functional.grid_sample
        padding_mode (str): Padding mode to be used for outside grid values. Default: 'zeros'.
            For more details check out docs of torch.nn.functional.grid_sample
        align_corners (bool): If True, consider -1 and 1 to refer to the centers of the corner pixels rather
            than the image corners. For more details check out docs of torch.nn.functional.grid_sample
            Default: True.

    Returns:
        Transformed image of the same shape as the input [B, C, S, S].
    """
    if len(image.shape) != 4:
        raise ValueError("Invalid image dimensions")

    if image.shape[-1] != image.shape[-2]:
        raise ValueError("Only square images are supported.")

    canvas_size = image.shape[-1]

    # Convert to 3x3 affine matrix.
    matrix = F.pad(matrix, (0, 0, 0, 1), "constant", value=0.0)
    matrix[:, -1, -1] = 1.0

    norm_m = torch.eye(3, device=image.device, dtype=torch.float32)
    norm_m[:2, 2] = -1.0
    norm_m[0, 0] *= 2 / (canvas_size - 1)
    norm_m[1, 1] *= 2 / (canvas_size - 1)
    norm_m = norm_m.unsqueeze(0)

    inv_norm_m = torch.linalg.inv_ex(norm_m)[0]

    # Normalize affine matrix since it's expected by the grid_sample function.
    matrix = norm_m @ (matrix @ inv_norm_m)

    matrix = torch.linalg.inv_ex(matrix)[0]
    grid = F.affine_grid(matrix[:, :2, :], image.shape, align_corners=True)

    return F.grid_sample(
        image,
        grid,
        mode=mode,
        padding_mode=padding_mode,
        align_corners=align_corners
    )


def get_convex_edges(polygons: Tensor, height: int, width: int) -> tuple[Tensor, Tensor]:
    """Gets the convex edges of the polygon.

    Edges are represented by the pair of x values (left and right) each having H values (one for each row).
    If x_left > x_right for some row, none of the pixels at that row is inside the polygon (whole polygon is either below
    or above that particular row).

    Args:
        polygons (Tensor): Represents polygons. Its shape is [B, N, 2].
            N is the number of polygon's vertices or edges and 2 is (x, y).
        height (int): Height of the image.
        width (int): Width of the image.

    Returns:
        The left and right edges of the convex polygon of shape [B, H].
    """
    device = polygons.device

    polygons = torch.cat([polygons, polygons[..., :1, :]], dim=-2)  # [B, N + 1, 2]
    x_start, y_start = polygons[..., :-1, 0], polygons[..., :-1, 1]  # [B, N]
    x_end, y_end = polygons[..., 1:, 0], polygons[..., 1:, 1]  # [B, N]

    # Calculate slope of each edge i.e. how much x passes if we change y by one (for the horizontal lines, we cap it at -w, w).
    dx = ((x_end - x_start) / (y_end - y_start + 1e-9)).clamp(-width, width)  # [B, N]

    ys = torch.arange(height, device=device, dtype=torch.float32).unsqueeze(-1)  # [H, 1]

    y_start = y_start.unsqueeze(1)  # [B, 1, N]
    y_end = y_end.unsqueeze(1)  # [B, 1, N]

    # Find out corresponding x-coordinate for each combination of edge and ypsilon in a range of [0, h).
    xs = (ys - y_start) * dx.unsqueeze(1) + x_start.unsqueeze(1)  # [B, H, N]

    # Compute valid points for each edge (the ones that are at the edges).
    valid_points = (y_start <= ys) & (ys <= y_end)  # [B, H, N]
    valid_points |= (y_start >= ys) & (ys >= y_end)  # [B, H, N]

    x_left_edges = torch.where(valid_points, xs, width)  # [B, H, N]
    x_right_edges = torch.where(valid_points, xs, -1)  # [B, H, N]

    # For every y find smallest (left) and largest (right) x value that is still at the edge of polygon.
    # Every point in between is inside convex polygon (this can be used e.g. to fill polygon with some color).
    x_left = x_left_edges.min(dim=-1).values
    x_right = x_right_edges.max(dim=-1).values
    return x_left, x_right


def draw_convex_polygon(images: Tensor, polygons: Tensor, colors: Tensor) -> Tensor:
    """Draws convex polygons on a batch of image tensors.

    Args:
        images (Tensor): Is tensor of shape [B, C, H, W].
        polygons (Tensor): Represents polygons as *ordered* vertices of shape [B, N, 2].
            N is the number of polygon's vertices or edges and 2 is (x, y).
        colors (Tensor): [B, C] or [B, 1] or [B] tensor.

    Returns:
        Drawn tensor with the same shape as the input [B, C, H, W].

    Note:
        This function assumes a coordinate system (0, h - 1), (0, w - 1) in the image, with (0, 0) being the center
        of the top-left pixel and (w - 1, h - 1) being the center of the bottom-right pixel.
    """
    batch_size, channels, height, width = images.shape
    device = images.device

    if colors.dim() == 1:
        if colors.shape[0] == 1:
            colors = colors.expand(batch_size)
            colors = colors.unsqueeze(1)
        elif colors.shape[0] == channels:
            colors = colors.unsqueeze(0)
        elif colors.shape[0] == batch_size:
            colors = colors.unsqueeze(1)

    x_left, x_right = get_convex_edges(polygons, height, width)  # [B, H]

    xs = torch.arange(width, device=device, dtype=torch.float32).view(1, 1, -1)  # [1, 1, W]
    fill_region = (xs >= x_left.unsqueeze(-1)) & (xs <= x_right.unsqueeze(-1))  # [B, H, W]

    fill_region = fill_region.unsqueeze(1)  # [B, 1, H, W]
    colors = colors.reshape(batch_size, -1, 1, 1).expand(batch_size, channels, 1, 1)  # [B, C, 1, 1]
    images = torch.where(fill_region, colors, images)  # [B, C, H, W]

    return images


def draw_non_convex_polygon(images: Tensor, polygons: Tensor, colors: Tensor) -> Tensor:
    """Draws nonconvex polygons on a batch of image tensors.

    Args:
        images (Tensor): Is tensor of shape [B, C, H, W].
        polygons (Tensor): Represents polygons as *ordered* vertices of shape [B, N, 2].
            N is the number of polygon's vertices or edges and 2 is (x, y).
        colors (Tensor): [B, C] or [B, 1] or [B] tensor by which each pixel of the polygon will be filled.

    Returns:
        Drawn tensor with the same shape as the input [B, C, H, W].
    """
    batch_size, channels, height, width = images.shape
    device = images.device
    dtype = polygons.dtype
    B, N, _ = polygons.shape

    if colors.dim() == 1:
        if colors.shape[0] == 1:
            colors = colors.view(1, 1, 1, 1).expand(batch_size, channels, 1, 1)
        elif colors.shape[0] == channels:
            colors = colors.view(1, channels, 1, 1).expand(batch_size, channels, 1, 1)
        elif colors.shape[0] == batch_size:
            colors = colors.view(batch_size, 1, 1, 1).expand(batch_size, channels, 1, 1)
    else:
        colors = colors.view(batch_size, -1, 1, 1).expand(batch_size, channels, 1, 1)

    v0 = polygons  # [B, N, 2]
    v1 = torch.roll(polygons, shifts=-1, dims=1)  # [B, N, 2]
    x0 = v0[..., 0]  # [B, N]
    y0 = v0[..., 1]
    x1 = v1[..., 0]
    y1 = v1[..., 1]

    edge_dy = y1 - y0  # [B, N]
    edge_dx = x1 - x0
    safe_edge_dy = torch.where(edge_dy == 0, torch.ones_like(edge_dy), edge_dy)
    slope = edge_dx / safe_edge_dy  # [B, N]

    py = torch.arange(height, device=device, dtype=dtype)

    # For each edge and each y-coordinate, compute if it crosses and where
    y0_exp = y0.unsqueeze(-1)  # [B, N, 1]
    y1_exp = y1.unsqueeze(-1)  # [B, N, 1]
    py_exp = py.view(1, 1, height)  # [1, 1, H]

    # Check if edge crosses the horizontal line at each y
    cond_cross = ((y0_exp <= py_exp) & (py_exp < y1_exp)) | ((y1_exp <= py_exp) & (py_exp < y0_exp))  # [B, N, H]

    # Compute x-intersection for each edge at each y-level: [B, N, H]
    x0_exp = x0.unsqueeze(-1)  # [B, N, 1]
    x_intersect = x0_exp + (py_exp - y0_exp) * slope.unsqueeze(-1)  # [B, N, H]

    # Mask out non-crossing edges with inf
    x_intersect = torch.where(cond_cross, x_intersect, torch.full_like(x_intersect, float('inf')))  # [B, N, H]

    # Sort intersections and use searchsorted to count crossings
    px = torch.arange(width, device=device, dtype=dtype)
    x_inter_flat = x_intersect.permute(0, 2, 1).reshape(batch_size * height, N)  # [B*H, N]
    x_sorted, _ = torch.sort(x_inter_flat, dim=1)  # [B*H, N]

    # For each x position, count how many intersections are to its left
    px_flat = px.view(1, width).expand(batch_size * height, width).contiguous()  # [B*H, W]
    crossing_counts = torch.searchsorted(x_sorted.contiguous(), px_flat, right=True)  # [B*H, W]

    # Reshape back to [B, H, W]
    crossing_counts = crossing_counts.view(batch_size, height, width)

    # Inside polygon if odd number of crossings
    fill_region = (crossing_counts % 2 == 1).unsqueeze(1)  # [B, 1, H, W]
    return torch.where(fill_region, colors, images)
