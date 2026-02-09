import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def linear_relu(in_dim: int, out_dim: int, add_layer_norm: bool = False, device: str = "cpu") -> nn.Module:
    """Creates a feed-forward block: Linear -> [LayerNorm] -> ReLU.

    Returns:
        nn.Sequential module that transforms input [B, in_dim] to [B, out_dim].
    """
    layers = [
        nn.Linear(in_dim, out_dim, device=device)
    ]

    if add_layer_norm:
        layers.append(nn.LayerNorm(out_dim, device=device))
    layers.append(nn.ReLU())

    return nn.Sequential(*layers)


def shuffle_upsample_conv_block(in_channels: int, out_channels: int, device: str = "cpu") -> nn.Module:
    """Creates a convolutional block that upsamples spatial resolution by 2x in both dimensions.

    Returns:
        nn.Module that outputs tensor of shape [B, out_channels, H*2, W*2] given input [B, in_channels, H, W].
    """
    return nn.Sequential(
        nn.Conv2d(in_channels, in_channels * 2, kernel_size=3, stride=1, padding=1, device=device),
        nn.ReLU(),
        nn.Conv2d(in_channels * 2, out_channels * 4, kernel_size=3, stride=1, padding=1, device=device),
        nn.PixelShuffle(2),
    )


class PixelShuffleNet(nn.Module):
    """
    MLP encoder with pixel shuffle decoder.

    This network is good with predicting shape mask of the brush stroke (1 channel output) but struggles with
    predicting color (use ModulatedPixelShuffleNet in those cases).

    Reference: https://arxiv.org/pdf/2011.08114
    """

    def __init__(
        self,
        input_dim: int,
        out_channels: int,
        add_layer_norm: bool = True,
        add_final_sigmoid: bool = False,
        device: str = "cpu",
    ):
        super().__init__()

        self.input_dim = input_dim
        self.add_final_sigmoid = add_final_sigmoid
        self.device = device

        self.encoder = nn.Sequential(
            linear_relu(input_dim, 512, device=device),
            linear_relu(512, 1024, device=device),
            linear_relu(1024, 2048, device=device),
            linear_relu(2048, 4096, add_layer_norm=add_layer_norm, device=device),
        )

        self.decoder = nn.Sequential(
            shuffle_upsample_conv_block(16, 8, device=device),
            shuffle_upsample_conv_block(8, 4, device=device),
            shuffle_upsample_conv_block(4, out_channels, device=device),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Predicts 2D output based on the given brush parameters.

        Args:
            x (Tensor): Batch brush params of shape [B, P].

        Returns:
             Tensor of shape [B, out_channels, 128, 128] with values in range [0, 1].
        """
        x = self.encoder(x)
        x = x.view(-1, 16, 16, 16)
        x = self.decoder(x)

        if self.add_final_sigmoid:
            x = F.sigmoid(x)

        return x


class PixelShuffleNetLight(nn.Module):
    """
    Light version of PixelShuffleNet for 32x32 output.

    Uses a smaller encoder (no 4096-dim layer) and single upsample block
    for faster training and inference with reduced memory footprint.

    Reference: Stylized Neural Painting (32x32 variant)
    """

    def __init__(
        self,
        input_dim: int,
        out_channels: int,
        add_layer_norm: bool = True,
        add_final_sigmoid: bool = False,
        device: str = "cpu",
    ):
        super().__init__()

        self.input_dim = input_dim
        self.add_final_sigmoid = add_final_sigmoid
        self.device = device

        self.encoder = nn.Sequential(
            linear_relu(input_dim, 512, device=device),
            linear_relu(512, 1024, device=device),
            linear_relu(1024, 2048, add_layer_norm=add_layer_norm, device=device),
        )

        self.decoder = shuffle_upsample_conv_block(8, out_channels, device=device)

    def forward(self, x: Tensor) -> Tensor:
        """Predicts 2D output based on the given brush parameters.

        Args:
            x (Tensor): Batch brush params of shape [B, P].

        Returns:
             Tensor of shape [B, out_channels, 32, 32].
        """
        x = self.encoder(x)
        x = x.view(-1, 8, 16, 16)
        x = self.decoder(x)

        if self.add_final_sigmoid:
            x = F.sigmoid(x)

        return x


class ModulatedShuffleUpsampleBlock(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, in_cond_dim: int, device: str = "cpu"):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.device = device

        self.upsample = shuffle_upsample_conv_block(in_channels, out_channels, device)
        self.film_gen = nn.Sequential(
            nn.Linear(in_cond_dim, 128, device=device),
            nn.ReLU(),
            nn.Linear(128, out_channels * 2, device=device)
        )

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        x = self.upsample(x)

        # Modulate each channel based on cond.
        gamma, beta = self.film_gen(cond).split([self.out_channels, self.out_channels], dim=1)  # Each: [B, out_channels]
        gamma = gamma.view(-1, self.out_channels, 1, 1)
        beta = beta.view(-1, self.out_channels, 1, 1)
        x = gamma * x + beta

        return x


class ModulatedConvBlock(nn.Module):
    """Modulated conv block without spatial upsampling.

    Processes features with FiLM conditioning but maintains spatial dimensions.
    """

    def __init__(self, in_channels: int, out_channels: int, in_cond_dim: int, device: str = "cpu"):
        super().__init__()

        self.out_channels = out_channels

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * 2, kernel_size=3, stride=1, padding=1, device=device),
            nn.ReLU(),
            nn.Conv2d(in_channels * 2, out_channels, kernel_size=3, stride=1, padding=1, device=device),
        )
        self.film_gen = nn.Sequential(
            nn.Linear(in_cond_dim, 128, device=device),
            nn.ReLU(),
            nn.Linear(128, out_channels * 2, device=device)
        )

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        x = self.conv(x)
        gamma, beta = self.film_gen(cond).split([self.out_channels, self.out_channels], dim=1)
        gamma = gamma.view(-1, self.out_channels, 1, 1)
        beta = beta.view(-1, self.out_channels, 1, 1)
        return gamma * x + beta


class ModulatedPixelShuffleNet(nn.Module):
    """
    MLP encoder with pixel shuffle decoder + channel modulation.

    This network is designed to capture complex color representations, addressing a key limitation of PixelShuffleNet,
    which is good at shape prediction but struggles with producing colored output.

    In addition to the standard pixel shuffle block, this network modulates channels using input brush parameters.

    References:
     * PixelShuffleNet (https://arxiv.org/pdf/2011.08114)
     * FiLM: Visual Reasoning with a General Conditioning Layer (https://arxiv.org/pdf/1709.07871)
    """

    def __init__(
        self,
        input_dim: int,
        out_channels: int,
        add_layer_norm: bool = True,
        device: str = "cpu",
    ):
        super().__init__()

        self.input_dim = input_dim
        self.device = device

        self.encoder = nn.Sequential(
            linear_relu(input_dim, 512, device=device),
            linear_relu(512, 1024, device=device),
            linear_relu(1024, 2048, device=device),
            linear_relu(2048, 4096, add_layer_norm=add_layer_norm, device=device),
        )

        self.upsample1 = ModulatedShuffleUpsampleBlock(16, 8, input_dim, device=device)
        self.upsample2 = ModulatedShuffleUpsampleBlock(8, 4, input_dim, device=device)

        self.out_upsample = shuffle_upsample_conv_block(4, out_channels, device)

    def forward(self, x: Tensor) -> Tensor:
        """Predicts 2D output based on the given brush parameters.

        Args:
            x (Tensor): Batch brush params of shape [B, P].

        Returns:
             Tensor of shape [B, out_channels, 128, 128] with values in range [0, 1].
        """
        encoded_x = self.encoder(x)

        img = encoded_x.view(-1, 16, 16, 16)

        # We're always conditioning by the input brush parameters.
        img = self.upsample1(img, x)
        img = self.upsample2(img, x)

        out = self.out_upsample(img)
        out = F.sigmoid(out)

        return out


class ModulatedPixelShuffleNetLight(nn.Module):
    """
    Light version of ModulatedPixelShuffleNet for 32x32 output.
    """

    def __init__(
        self,
        input_dim: int,
        out_channels: int,
        add_layer_norm: bool = True,
        device: str = "cpu",
    ):
        super().__init__()

        self.input_dim = input_dim
        self.device = device

        self.encoder = nn.Sequential(
            linear_relu(input_dim, 512, device=device),
            linear_relu(512, 1024, device=device),
            linear_relu(1024, 2048, add_layer_norm=add_layer_norm, device=device),
        )

        # Modulated processing block (no upsampling).
        self.process = ModulatedConvBlock(8, 8, input_dim, device=device)

        # Upsampling block (16x16 → 32x32)
        self.out_upsample = ModulatedShuffleUpsampleBlock(8, out_channels, input_dim, device=device)

    def forward(self, x: Tensor) -> Tensor:
        """Predicts 2D output based on the given brush parameters.

        Args:
            x (Tensor): Batch brush params of shape [B, P].

        Returns:
             Tensor of shape [B, out_channels, 32, 32] with values in range [0, 1].
        """
        encoded_x = self.encoder(x)
        img = encoded_x.view(-1, 8, 16, 16)

        img = self.process(img, x)

        out = self.out_upsample(img, x)
        out = F.sigmoid(out)

        return out


class ConvTransposeDecoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        out_channels: int,
        ngf: int = 64,
        device: str = "cpu",
    ):
        super().__init__()

        self.out_size = 128
        self.main = nn.Sequential(
            nn.ConvTranspose2d(input_dim, ngf * 8, 4, 1, 0, bias=False, device=device),
            nn.BatchNorm2d(ngf * 8, device=device),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 8, ngf * 8, 4, 2, 1, bias=False, device=device),
            nn.BatchNorm2d(ngf * 8, device=device),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False, device=device),
            nn.BatchNorm2d(ngf * 4, device=device),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False, device=device),
            nn.BatchNorm2d(ngf * 2, device=device),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False, device=device),
            nn.BatchNorm2d(ngf, device=device),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf, out_channels, 4, 2, 1, bias=False, device=device),
        )

    def forward(self, input):
        out = self.main(input)
        return out


class ConvTransposeDecoderLight(nn.Module):
    """
    Light version of ConvTransposeDecoder for 32x32 output.

    Uses 4 ConvTranspose2d layers instead of 6: 1→4→8→16→32 spatial resolution.
    """

    def __init__(
        self,
        input_dim: int,
        out_channels: int,
        ngf: int = 64,
        device: str = "cpu",
    ):
        super().__init__()

        self.out_size = 32
        self.main = nn.Sequential(
            nn.ConvTranspose2d(input_dim, ngf * 4, 4, 1, 0, bias=False, device=device),
            nn.BatchNorm2d(ngf * 4, device=device),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False, device=device),
            nn.BatchNorm2d(ngf * 2, device=device),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False, device=device),
            nn.BatchNorm2d(ngf, device=device),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf, out_channels, 4, 2, 1, bias=False, device=device),
        )

    def forward(self, input):
        out = self.main(input)
        return out


class ShadingRasterNetwork(nn.Module):
    """
    ZouFCNFusion-style network combining a PixelShuffle-style raster decoder for mask
    and a DCGAN-style shader for color. Architecture was taken almost directly from the
    https://github.com/jiupinjia/stylized-neural-painting/blob/main/networks.py with 2 modifications:
    * Shader returns 3 output channels instead of 6.
    * Raster returns 2 output channel instead of just one. One represents mask for the shader and one represents alpha mask.
      This was done so that single masks wouldn't compete with loss from alpha and loss from foreground.
    * Alpha mask has 1 output channel and is directly equal to the output of raster network without multiplying by brush param alpha.
      This was done because not every brush stroke has alpha param and also, relationship between alpha and final alpha mask
      might be more complex than just simple multiplication.
    """

    def __init__(
        self,
        shading_input_indices: list[int],
        raster_input_indices: list[int],
        device: str = "cpu",
    ):
        super().__init__()
        self.shading_input_indices = shading_input_indices
        self.raster_input_indices = raster_input_indices
        self.device = device

        self.raster_encoder = nn.Sequential(
            nn.Linear(len(self.raster_input_indices), 512, device=device),
            nn.ReLU(),
            nn.Linear(512, 1024, device=device),
            nn.ReLU(),
            nn.Linear(1024, 2048, device=device),
            nn.ReLU(),
            nn.Linear(2048, 4096, device=device),
            nn.ReLU(),
        )
        self.raster_decoder = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1, device=device),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, device=device),
            nn.PixelShuffle(upscale_factor=2),

            nn.Conv2d(8, 16, kernel_size=3, padding=1, device=device),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, padding=1, device=device),
            nn.PixelShuffle(upscale_factor=2),

            nn.Conv2d(4, 8, kernel_size=3, padding=1, device=device),
            nn.ReLU(),
            nn.Conv2d(8, 8, kernel_size=3, padding=1, device=device),
            nn.PixelShuffle(upscale_factor=2),
        )

        ngf = 64
        self.shader = nn.Sequential(
            nn.ConvTranspose2d(len(self.shading_input_indices), ngf * 8, kernel_size=4, bias=False, device=device),
            nn.BatchNorm2d(ngf * 8, device=device),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(ngf * 8, ngf * 8, kernel_size=4, stride=2, padding=1, bias=False, device=device),
            nn.BatchNorm2d(ngf * 8, device=device),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(ngf * 8, ngf * 4, kernel_size=4, stride=2, padding=1, bias=False, device=device),
            nn.BatchNorm2d(ngf * 4, device=device),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(ngf * 4, ngf * 2, kernel_size=4, stride=2, padding=1, bias=False, device=device),
            nn.BatchNorm2d(ngf * 2, device=device),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(ngf * 2, ngf, kernel_size=4, stride=2, padding=1, bias=False, device=device),
            nn.BatchNorm2d(ngf, device=device),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(ngf, 3, kernel_size=4, stride=2, padding=1, bias=False, device=device),
        )

    def forward(self, input: Tensor) -> Tensor:
        """
        Forward pass combining mask and color predictions.

        Args:
            input: Input parameters [B, input_dim]

        Returns:
            RGBA tensor [B, 4, 128, 128] where RGB = color * mask
        """
        raster_params = input[:, self.raster_input_indices]

        encoded_x = self.raster_encoder(raster_params)
        encoded_x = encoded_x.view(-1, 16, 16, 16)

        mask = self.raster_decoder(encoded_x)
        mask = mask.view(-1, 2, 128, 128)
        raster_mask, alpha_mask = mask.split([1, 1], dim=1)

        # We pass each param as a separate 2D plane into the shader.
        shader_params = input[:, self.shading_input_indices, None, None]
        color = self.shader(shader_params)
        foreground = color * raster_mask

        return torch.cat([foreground, alpha_mask], dim=1)


class ShadingRasterNetworkLight(nn.Module):
    """
    Light version of ShadingRasterNetwork for 32x32 output.
    """

    def __init__(
        self,
        shading_input_indices: list[int],
        raster_input_indices: list[int],
        device: str = "cpu",
    ):
        super().__init__()
        self.shading_input_indices = shading_input_indices
        self.raster_input_indices = raster_input_indices
        self.device = device

        self.raster_encoder = nn.Sequential(
            nn.Linear(len(self.raster_input_indices), 512, device=device),
            nn.ReLU(),
            nn.Linear(512, 1024, device=device),
            nn.ReLU(),
            nn.Linear(1024, 2048, device=device),
            nn.ReLU(),
        )
        self.raster_decoder = nn.Sequential(
            nn.Conv2d(8, 64, kernel_size=3, padding=1, device=device),
            nn.ReLU(),
            nn.Conv2d(64, 8, kernel_size=3, padding=1, device=device),
            nn.PixelShuffle(upscale_factor=2),
        )

        ngf = 64
        self.shader = nn.Sequential(
            nn.ConvTranspose2d(len(shading_input_indices), ngf * 8, kernel_size=4, bias=False, device=device),
            nn.BatchNorm2d(ngf * 8, device=device),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(ngf * 8, ngf * 4, kernel_size=4, stride=2, padding=1, bias=False, device=device),
            nn.BatchNorm2d(ngf * 4, device=device),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(ngf * 4, ngf * 2, kernel_size=4, stride=2, padding=1, bias=False, device=device),
            nn.BatchNorm2d(ngf * 2, device=device),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(ngf * 2, 3, kernel_size=4, stride=2, padding=1, bias=False, device=device),
        )

    def forward(self, input: Tensor) -> Tensor:
        """
        Forward pass combining mask and color predictions.

        Args:
            input: Input parameters [B, input_dim]

        Returns:
            RGBA tensor [B, 4, 32, 32] where RGB = color * mask
        """
        raster_params = input[:, self.raster_input_indices]

        encoded_x = self.raster_encoder(raster_params)
        encoded_x = encoded_x.view(-1, 8, 16, 16)

        mask = self.raster_decoder(encoded_x)
        mask = mask.view(-1, 2, 32, 32)
        raster_mask, alpha_mask = mask.split([1, 1], dim=1)

        # We pass each param as a separate 2D plane into the shader.
        shader_params = input[:, self.shading_input_indices, None, None]
        color = self.shader(shader_params)

        foreground = color * raster_mask

        return torch.cat([foreground, alpha_mask], dim=1)
