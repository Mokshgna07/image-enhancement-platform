from __future__ import annotations

import torch
from torch import nn


class ResidualBlock(nn.Module):
    """
    EDSR-inspired residual block.

    The block learns a residual mapping:

        y = x + F(x)

    where F(x) consists of two convolution layers
    with a ReLU activation between them.
    """

    def __init__(
        self,
        channels: int,
        residual_scale: float = 1.0,
    ):
        super().__init__()

        self.residual_scale = residual_scale

        self.conv1 = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.activation = nn.ReLU(
            inplace=True
        )

        self.conv2 = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        residual = self.conv1(x)

        residual = self.activation(
            residual
        )

        residual = self.conv2(
            residual
        )

        return (
            x
            + self.residual_scale * residual
        )
