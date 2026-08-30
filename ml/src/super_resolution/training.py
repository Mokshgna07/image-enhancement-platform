from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """
    Set random seeds for reproducibility.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(
    requested: str = "auto",
) -> torch.device:

    requested = requested.lower()

    if requested == "cuda":

        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was explicitly requested "
                "but is not available."
            )

        return torch.device("cuda")

    if requested == "cpu":
        return torch.device("cpu")

    if requested == "auto":

        if torch.cuda.is_available():
            return torch.device("cuda")

        return torch.device("cpu")

    raise ValueError(
        f"Unsupported device: {requested}"
    )


class JSONLLogger:
    """
    Append training metrics as JSON objects.

    Each line represents one logging event.
    """

    def __init__(
        self,
        path: str | Path,
    ):
        self.path = Path(path)

        self.path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

    def log(
        self,
        data: dict[str, Any],
    ) -> None:

        with self.path.open(
            "a",
            encoding="utf-8",
        ) as file:

            file.write(
                json.dumps(
                    data,
                    default=str,
                )
                + "\n"
            )


class EarlyStopping:
    def __init__(
        self,
        patience: int,
        min_delta: float = 0.0,
    ):
        self.patience = patience
        self.min_delta = min_delta

        self.best = None
        self.bad_epochs = 0

    def step(
        self,
        value: float,
    ) -> bool:

        if self.best is None:

            self.best = value
            self.bad_epochs = 0

            return False

        if value > self.best + self.min_delta:

            self.best = value
            self.bad_epochs = 0

            return False

        self.bad_epochs += 1

        return (
            self.bad_epochs
            >= self.patience
        )
