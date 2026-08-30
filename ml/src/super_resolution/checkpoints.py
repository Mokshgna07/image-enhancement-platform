from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn


class CheckpointManager:

    def __init__(
        self,
        directory: str | Path,
    ):
        self.directory = Path(
            directory
        )

        self.directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    def save(
        self,
        filename: str,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        epoch: int,
        best_metric: float,
        config: dict,
        metadata: dict,
        scaler: Any | None = None,
    ) -> Path:

        path = (
            self.directory
            / filename
        )

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": (
                model.state_dict()
            ),
            "optimizer_state_dict": (
                optimizer.state_dict()
            ),
            "scheduler_state_dict": (
                scheduler.state_dict()
                if scheduler is not None
                else None
            ),
            "best_metric": best_metric,
            "config": config,
            "metadata": metadata,
        }

        if scaler is not None:
            checkpoint[
                "scaler_state_dict"
            ] = scaler.state_dict()

        torch.save(
            checkpoint,
            path,
        )

        return path

    @staticmethod
    def load(
        path: str | Path,
        model: nn.Module,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: Any | None = None,
        scaler: Any | None = None,
        device: str | torch.device = "cpu",
    ) -> dict:

        checkpoint = torch.load(
    path,
    map_location=device,
    weights_only=False,
)

        model.load_state_dict(
            checkpoint[
                "model_state_dict"
            ]
        )

        if (
            optimizer is not None
            and checkpoint.get(
                "optimizer_state_dict"
            )
            is not None
        ):
            optimizer.load_state_dict(
                checkpoint[
                    "optimizer_state_dict"
                ]
            )

        if (
            scheduler is not None
            and checkpoint.get(
                "scheduler_state_dict"
            )
            is not None
        ):
            scheduler.load_state_dict(
                checkpoint[
                    "scheduler_state_dict"
                ]
            )

        if (
            scaler is not None
            and checkpoint.get(
                "scaler_state_dict"
            )
            is not None
        ):
            scaler.load_state_dict(
                checkpoint[
                    "scaler_state_dict"
                ]
            )

        return checkpoint
