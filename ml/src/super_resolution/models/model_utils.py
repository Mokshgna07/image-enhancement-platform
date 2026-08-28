from __future__ import annotations

from pathlib import Path

import torch
from torch import nn


def count_parameters(
    model: nn.Module,
    trainable_only: bool = False,
) -> int:

    parameters = model.parameters()

    if trainable_only:
        parameters = (
            parameter
            for parameter in parameters
            if parameter.requires_grad
        )

    return sum(
        parameter.numel()
        for parameter in parameters
    )


def model_summary(
    model: nn.Module,
) -> str:

    total = count_parameters(
        model
    )

    trainable = count_parameters(
        model,
        trainable_only=True,
    )

    lines = [
        f"Model: {model.__class__.__name__}",
        f"Total parameters: {total:,}",
        f"Trainable parameters: {trainable:,}",
    ]

    return "\n".join(lines)


def save_model(
    model: nn.Module,
    path: str | Path,
) -> None:

    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        {
            "model_state_dict": (
                model.state_dict()
            ),
        },
        path,
    )


def load_model(
    model: nn.Module,
    path: str | Path,
    device: str | torch.device = "cpu",
) -> nn.Module:

    checkpoint = torch.load(
        path,
        map_location=device,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    return model
