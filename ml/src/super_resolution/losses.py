from __future__ import annotations

import torch
from torch import nn


class CharbonnierLoss(nn.Module):
    """
    Charbonnier loss.

    A smooth approximation of L1 loss:

        sqrt((prediction - target)^2 + epsilon^2)
    """

    def __init__(
        self,
        epsilon: float = 1e-6,
    ):
        super().__init__()

        self.epsilon = epsilon

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:

        difference = prediction - target

        loss = torch.sqrt(
            difference.pow(2)
            + self.epsilon ** 2
        )

        return loss.mean()


class L1Loss(nn.Module):
    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:

        return torch.abs(
            prediction - target
        ).mean()


class MSELoss(nn.Module):
    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:

        return torch.mean(
            (prediction - target).pow(2)
        )


def build_loss(
    loss_config: dict,
) -> nn.Module:

    loss_type = loss_config["type"].lower()

    if loss_type == "charbonnier":
        return CharbonnierLoss(
            epsilon=float(
                loss_config.get(
                    "epsilon",
                    1e-6,
                )
            )
        )

    if loss_type == "l1":
        return L1Loss()

    if loss_type in {"mse", "l2"}:
        return MSELoss()

    raise ValueError(
        f"Unsupported loss type: {loss_type}"
    )
