from __future__ import annotations

import torch
from torch import nn

from .residual_block import ResidualBlock
from .upsampler import PixelShuffleBlock


class EDSR(nn.Module):
    """
    EDSR-inspired CNN for single-image super-resolution.

    Parameters
    ----------
    scale:
        Super-resolution scale factor. Supports 2 or 4.

    channels:
        Number of feature channels.

    num_blocks:
        Number of residual blocks.

    residual_scale:
        Scaling factor applied to each residual branch.
    """

    def __init__(
        self,
        scale: int = 4,
        channels: int = 64,
        num_blocks: int = 16,
        residual_scale: float = 0.1,
    ):
        super().__init__()

        if scale not in (2, 4):
            raise ValueError(
                "Only 2x and 4x super-resolution "
                "are supported."
            )

        if channels <= 0:
            raise ValueError(
                "channels must be positive."
            )

        if num_blocks <= 0:
            raise ValueError(
                "num_blocks must be positive."
            )

        self.scale = scale
        self.channels = channels
        self.num_blocks = num_blocks

        # Shallow feature extraction.
        self.head = nn.Conv2d(
            3,
            channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        # Deep residual feature extraction.
        self.body = nn.Sequential(
            *[
                ResidualBlock(
                    channels=channels,
                    residual_scale=residual_scale,
                )
                for _ in range(num_blocks)
            ],
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
        )

        # Learned spatial upsampling.
        self.upsampler = PixelShuffleBlock(
            channels=channels,
            scale=scale,
        )

        # RGB reconstruction.
        self.tail = nn.Conv2d(
            channels,
            3,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """
        Initialize convolution weights using Kaiming
        initialization, suitable for ReLU-based networks.
        """

        for module in self.modules():

            if isinstance(
                module,
                nn.Conv2d,
            ):
                nn.init.kaiming_normal_(
                    module.weight,
                    mode="fan_out",
                    nonlinearity="relu",
                )

                if module.bias is not None:
                    nn.init.zeros_(
                        module.bias
                    )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        shallow = self.head(x)

        body = self.body(
            shallow
        )

        # Global residual connection.
        body = body + shallow

        upsampled = self.upsampler(
            body
        )

        output = self.tail(
            upsampled
        )

        return output
