from __future__ import annotations

import torch
from torch import nn


class PixelShuffleBlock(nn.Module):
    """
    Upsampling block using sub-pixel convolution.

    A convolution first produces:

        channels × scale²

    feature channels.

    PixelShuffle then rearranges those channels into
    spatial resolution.
    """

    def __init__(
        self,
        channels: int,
        scale: int,
    ):
        super().__init__()

        if scale not in (2, 4):
            raise ValueError(
                "Scale must be 2 or 4."
            )

        layers: list[nn.Module] = []

        if scale == 4:
            scales = [2, 2]
        else:
            scales = [2]

        for current_scale in scales:

            layers.append(
                nn.Conv2d(
                    channels,
                    channels
                    * current_scale
                    * current_scale,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                )
            )

            layers.append(
                nn.PixelShuffle(
                    current_scale
                )
            )

            layers.append(
                nn.ReLU(
                    inplace=True
                )
            )

        self.body = nn.Sequential(
            *layers
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        return self.body(x)
