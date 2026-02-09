import random
from abc import ABC, abstractmethod

import kornia.geometry.transform as kornia_transform
import torch
import torch.nn as nn
from torch import Tensor
from torchvision import models


class ImageLoss(nn.Module, ABC):
    """Abstract base class for image-based loss functions.

    All ImageLoss implementations expect RGB images with values in range [0, 255].
    Normalization to [0, 1] (or other ranges) is handled internally by each implementation.
    """

    @abstractmethod
    def forward(self, img_pred: Tensor, img_target: Tensor) -> Tensor:
        """Computes image loss.

        Args:
            img_pred (Tensor): RGB images of shape [B, 3, H, W] with values in range [0, 255].
            img_target (Tensor): RGB images of shape [B, 3, H, W] with values in range [0, 255].

        Returns:
            Tensor of shape [B] with computed loss for each input pair.
        """
        pass


class CombinedImageLoss(ImageLoss):

    def __init__(
        self,
        losses: list[ImageLoss],
        weights: list[float] | None = None,
        normalize_by_weight_sum: bool = True,
    ):
        super().__init__()

        if weights is None:
            weights = [1.0] * len(losses)

        if len(weights) != len(losses):
            raise ValueError(
                f"weights length ({len(weights)}) must match losses length ({len(losses)})"
            )

        for i, w in enumerate(weights):
            if w <= 0:
                raise ValueError(f"weights[{i}] must be > 0, got {w}")

        self.losses = nn.ModuleList(losses)
        self.weights = weights
        self.normalize_by_weight_sum = normalize_by_weight_sum

    def forward(self, img_pred: Tensor, img_target: Tensor) -> Tensor:
        """Compute weighted combination of all component losses.

        Args:
            img_pred: Predicted RGB images [B, 3, H, W] in range [0, 255].
            img_target: Target RGB images [B, 3, H, W] in range [0, 255].

        Returns:
            Combined loss tensor of shape [B].
        """
        total = 0

        for loss_fn, w in zip(self.losses, self.weights):
            total = total + w * loss_fn(img_pred, img_target)

        if self.normalize_by_weight_sum:
            total = total / sum(self.weights)

        return total


class SinkhornLoss(ImageLoss):
    """Earth mover's distance loss using Sinkhorn algorithm.

    Pixel values are treated as probability masses and normalized internally to create distributions.

    This loss provides non-zero gradient signal when trying to optimize brush stroke parameters
    towards some other brush stroke, even when there is no overlap between them on the drawn canvas.

    For more details, check out section 3.3. in paper https://arxiv.org/pdf/2011.08114.

    Implementation was modified from the following sources:
    * https://github.com/fwilliams/scalable-pytorch-sinkhorn/blob/main/sinkhorn.py
    * https://github.com/jiupinjia/stylized-neural-painting/blob/main/loss.py#L128
    """

    def __init__(
        self,
        sinkhorn_canvas_size: int = 48,
        iterations: int = 5,
        eps: float = 0.01,
        device: str = "cpu"
    ):
        """Initialize SinkhornLoss.

        Args:
            sinkhorn_canvas_size: Canvas size for Sinkhorn computation. Images are resized to this size. Default: 48.
            iterations: Number of Sinkhorn iterations. Default: 5.
            eps: Regularization parameter for entropy. Lower values give sharper transport. Default: 0.01.
            device: Device for PyTorch tensors. Default: 'cpu'.
        """
        super().__init__()
        self.sinkhorn_canvas_size = sinkhorn_canvas_size
        self.iterations = iterations
        self.eps = eps
        self.device = device

        # Compute normalized cost matrix of shape [size * size, size * size] between pixels.
        # The [0, 0] value means dist([0, 0], [0, 0]).
        # The [0, size * size] means dist([0, 0], [size, size]).
        size = sinkhorn_canvas_size
        y, x = torch.meshgrid(torch.arange(size, device=device), torch.arange(size, device=device), indexing="ij")
        pixels = torch.stack((y.flatten(), x.flatten()), dim=1)  # Shape [size ** 2, 2]
        pixels = pixels / size
        cost_matrix = (pixels.unsqueeze(1) - pixels.unsqueeze(0)).pow(2).sum(dim=-1)  # Shape [size ** 2, size ** 2]

        assert cost_matrix.shape[-1] == cost_matrix.shape[-2] == size ** 2
        self.register_buffer("cost_matrix", cost_matrix)

    def forward(self, img_pred: Tensor, img_target: Tensor) -> Tensor:
        batch_size = img_pred.shape[0]
        shape = img_pred.shape
        canvas_size = shape[-1]

        if len(img_pred.shape) != 4 or len(img_target.shape) != 4:
            raise ValueError(
                "Only batched 4D input is supported."
            )

        if img_pred.shape != img_target.shape:
            raise ValueError(
                f"Shapes of the input images must be equal but shapes {img_pred.shape, img_target.shape} were passed"
            )

        if shape[-1] != shape[-2]:
            raise ValueError(f"Only square images are supported. Input image shape: {shape}")

        if canvas_size < self.sinkhorn_canvas_size:
            raise ValueError(
                f"Input image size must be higher or equal to {self.sinkhorn_canvas_size} "
                f"as was configured in the constructor"
            )

        if canvas_size > self.sinkhorn_canvas_size:
            img_pred = nn.functional.interpolate(img_pred, self.sinkhorn_canvas_size, mode="area")
            img_target = nn.functional.interpolate(img_target, self.sinkhorn_canvas_size, mode="area")

        # Pick just one random channel as they did in the Stylized Neural Painting paper.
        rand_index = random.randint(0, img_pred.shape[1] - 1)
        img_pred = img_pred[:, rand_index].reshape(batch_size, -1)  # [B, S * S]
        img_target = img_target[:, rand_index].reshape(batch_size, -1)  # [B, S * S]

        # Normalize to create distribution mass (add small constant to prevent division by zero).
        img_pred = img_pred + 1e-9
        img_target = img_target + 1e-9
        mass_source = img_pred / img_pred.sum(dim=-1, keepdim=True)  # [B, S * S]
        mass_target = img_target / img_target.sum(dim=-1, keepdim=True)  # [B, S * S]

        # Sinkhorn loop, take from https://github.com/jiupinjia/stylized-neural-painting/blob/main/pytorch_batch_sinkhorn.py#L63.
        log_u = mass_source.log()
        log_v = mass_target.log()
        u = torch.zeros_like(mass_source)
        v = torch.zeros_like(mass_target)

        def M(u, v):
            """Modified cost for logarithmic updates"""
            return (-self.cost_matrix + u.unsqueeze(2) + v.unsqueeze(1)) / self.eps

        def lse(A):
            """log-sum-exp"""
            return torch.log(torch.exp(A).sum(2, keepdim=True) + 1e-6)

        for _ in range(self.iterations):
            u = self.eps * (log_u - lse(M(u, v)).squeeze()) + u
            v = self.eps * (log_v - lse(M(u, v).transpose(dim0=1, dim1=2)).squeeze()) + v

        P_ij = torch.exp(M(u, v))
        cost = torch.sum(P_ij * self.cost_matrix, dim=[1, 2])

        return cost


class PixelLoss(ImageLoss):
    """Standard L^p pixel-wise loss between images."""

    def __init__(self, p: int = 1, ignore_color: bool = False):
        """Initialize PixelLoss.

        Args:
            p: Power for L^p norm (1 for L1, 2 for L2). Default: 1.
            ignore_color: If True, convert to grayscale before computing loss. Default: False.
        """
        super().__init__()
        self.p = p
        self.ignore_color = ignore_color

    def forward(self, img_pred: Tensor, img_target: Tensor) -> Tensor:
        if self.ignore_color:
            img_pred = img_pred.mean(dim=1, keepdim=True)
            img_target = img_target.mean(dim=1, keepdim=True)

        img_pred = img_pred / 255.0
        img_target = img_target / 255.0

        loss = (img_target - img_pred).abs().pow(self.p).mean(dim=[1, 2, 3])
        return loss


class GatysStyleLoss(ImageLoss):
    """Loss similar to Gatys neural style loss using VGG16 Gram matrices.

    This loss measures style similarity by comparing Gram matrices (feature correlations)
    between predicted images and a reference style image. The Gram matrix captures
    texture and color patterns without spatial information, making it ideal for style transfer.

    Based on: "A Neural Algorithm of Artistic Style" (Gatys et al., 2015)
    https://arxiv.org/abs/1508.06576
    """

    LAYER_MAP = {
        "relu1_2": 3, "relu2_2": 8, "relu3_3": 15, "relu4_3": 22, "relu5_3": 29,
    }

    # ImageNet normalization constants
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    def __init__(
        self,
        style_layers: list[str] | None = None,
        style_layer_weights: list[float] | None = None,
        device: str = "cpu",
    ) -> None:
        """Initialize Gatys style loss with Gram matrix computation.

        Args:
            style_layers (List[str] | None): VGG16 layers for style extraction.
                Defaults to ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3'].
                Earlier layers capture low-level textures/colors, later layers capture higher-level patterns.
            style_layer_weights (List[float] | None): Per-layer loss weights.
                Defaults to uniform weights [1, 4, 16, 64]. Must have same length as style_layers.
            device (str): Device for computation ('cpu' or 'cuda'). Default: 'cpu'.

        Raises:
            ValueError: If style_weights length doesn't match style_layers length.
            ValueError: If unknown layer name is provided.
        """
        super().__init__()
        self.device = device

        if style_layers is None:
            style_layers = ["relu1_2", "relu2_2", "relu3_3", "relu4_3"]

        if style_layer_weights is None:
            style_layer_weights = [1, 4, 16, 64]

        if len(style_layer_weights) != len(style_layers):
            raise ValueError(
                f"style_layer_weights length ({len(style_layer_weights)}) must match style_layers length ({len(style_layers)})")

        for layer in style_layers:
            if layer not in self.LAYER_MAP:
                raise ValueError(f"Unknown layer '{layer}'. Valid layers: {list(self.LAYER_MAP.keys())}")

        self.style_layers = style_layers
        self.style_layer_weights = style_layer_weights

        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features.to(device)
        for i, layer in enumerate(vgg):
            if isinstance(layer, torch.nn.MaxPool2d):
                vgg[i] = torch.nn.AvgPool2d(kernel_size=2, stride=2, padding=0)

        # Freeze weights.
        vgg.eval()
        for param in vgg.parameters():
            param.requires_grad = False
        self.vgg = vgg

        self.register_buffer("mean", torch.tensor(self.IMAGENET_MEAN, device=device).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(self.IMAGENET_STD, device=device).view(1, 3, 1, 1))

    @staticmethod
    def create(color_only: bool = False, device: str = "cpu") -> "GatysStyleLoss":
        """Factory method to create a GatysStyleLoss with preset layer configurations.

        Args:
            color_only (bool): If True, use only early layers (conv1, conv2) for color transfer.
                If False, use conv1-4 for full color and texture transfer. Default: False.
            device (str): Device for computation ('cpu' or 'cuda'). Default: 'cpu'.

        Returns:
            GatysStyleLoss: Configured style loss instance.
        """
        if color_only:
            style_layers = ["relu1_2", "relu2_2"]
            style_layer_weights = [1, 4]
        else:
            style_layers = ["relu1_2", "relu2_2", "relu3_3", "relu4_3"]
            style_layer_weights = [1, 4, 16, 64]

        return GatysStyleLoss(style_layers=style_layers, style_layer_weights=style_layer_weights, device=device)

    def forward(self, img_pred: Tensor, img_style: Tensor) -> Tensor:
        """Compute style loss between predicted image and style reference."""
        B = img_pred.shape[0]

        img_pred = img_pred / 255.0
        img_style = img_style / 255.0

        pred_features = self._extract_features(img_pred)
        style_features = self._extract_features(img_style)

        loss = torch.zeros(B, device=img_pred.device, dtype=torch.float32)

        for layer_name, weight in zip(self.style_layers, self.style_layer_weights):
            gram_pred = self._gram_matrix(pred_features[layer_name])  # [B, C, C]
            gram_style = self._gram_matrix(style_features[layer_name])  # [B, C, C]

            layer_loss = (gram_pred - gram_style).pow(2).mean(dim=(1, 2))  # [B]

            loss = loss + weight * layer_loss

        loss = loss / sum(self.style_layer_weights)

        return loss

    def _extract_features(self, img: Tensor) -> dict[str, Tensor]:
        """Extract VGG16 features from specified style layers.

        Args:
            img (Tensor): Input image [B, C, H, W] in range [0, 1] (already normalized).

        Returns:
            Dict[str, Tensor]: Dictionary mapping layer names to their feature tensors.
        """
        img = self._preprocess(img)
        features = {}

        max_layer_idx = max(self.LAYER_MAP[layer] for layer in self.style_layers)

        x = img
        for idx, layer in enumerate(self.vgg):
            x = layer(x)
            for layer_name in self.style_layers:
                if self.LAYER_MAP[layer_name] == idx:
                    features[layer_name] = x

            if idx >= max_layer_idx:
                break

        return features

    def _preprocess(self, img: Tensor) -> Tensor:
        """Preprocess image for VGG16."""
        img = kornia_transform.resize(img, interpolation="bilinear", size=(224, 224), align_corners=False)
        img = (img - self.mean) / self.std

        return img

    @staticmethod
    def _gram_matrix(features: Tensor) -> Tensor:
        """Compute normalized Gram matrix (feature correlations).

        The Gram matrix G_ij represents the correlation between feature maps i and j,
        capturing texture information while discarding spatial layout.

        Args:
            features (Tensor): Feature maps [B, C, H, W].

        Returns:
            Tensor: Normalized Gram matrix [B, C, C].
        """
        B, C, H, W = features.shape
        F = features.view(B, C, H * W)
        G = torch.bmm(F, F.transpose(1, 2))
        return G / (C * H * W)
