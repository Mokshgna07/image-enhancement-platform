
from __future__ import annotations

import torch
from torch import nn


class ResidualBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        residual_scale: float = 0.1,
    ):
        super().__init__()

        self.residual_scale = residual_scale

        self.conv1 = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
        )

        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.conv1(x)
        residual = self.relu(residual)
        residual = self.conv2(residual)

        return x + residual * self.residual_scale


class Upsampler(nn.Module):
    def __init__(
        self,
        scale: int,
        channels: int,
    ):
        super().__init__()

        layers = []

        if scale == 4:
            factors = [2, 2]
        elif scale == 2:
            factors = [2]
        else:
            raise ValueError(
                "EDSR supports scale factors 2 and 4."
            )

        for factor in factors:
            layers.append(
                nn.Conv2d(
                    channels,
                    channels * factor * factor,
                    kernel_size=3,
                    padding=1,
                )
            )

            layers.append(
                nn.PixelShuffle(factor)
            )

            layers.append(
                nn.ReLU(inplace=True)
            )

        self.body = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class EDSR(nn.Module):
    def __init__(
        self,
        scale: int = 4,
        channels: int = 64,
        num_blocks: int = 16,
        residual_scale: float = 0.1,
    ):
        super().__init__()

        self.scale = scale

        self.head = nn.Conv2d(
            3,
            channels,
            kernel_size=3,
            padding=1,
        )

        self.body = nn.Sequential(
            *[
                ResidualBlock(
                    channels,
                    residual_scale=residual_scale,
                )
                for _ in range(num_blocks)
            ],
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1,
            ),
        )

        self.upsampler = Upsampler(
            scale=scale,
            channels=channels,
        )

        self.tail = nn.Conv2d(
            channels,
            3,
            kernel_size=3,
            padding=1,
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight,
                    mode="fan_out",
                    nonlinearity="relu",
                )

                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shallow = self.head(x)

        body = self.body(shallow)

        body = body + shallow

        upsampled = self.upsampler(body)

        output = self.tail(upsampled)

        return output.clamp(0.0, 1.0)


def count_parameters(model: nn.Module) -> int:
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
