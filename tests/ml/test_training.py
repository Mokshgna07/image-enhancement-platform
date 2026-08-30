from pathlib import Path

import torch

from ml.src.super_resolution.checkpoints import (
    CheckpointManager,
)
from ml.src.super_resolution.losses import (
    CharbonnierLoss,
)
from ml.src.super_resolution.models import (
    EDSR,
)
from ml.src.super_resolution.training import (
    get_device,
)


def test_charbonnier_loss():

    criterion = CharbonnierLoss()

    prediction = torch.rand(
        2,
        3,
        16,
        16,
    )

    target = torch.rand(
        2,
        3,
        16,
        16,
    )

    loss = criterion(
        prediction,
        target,
    )

    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_device_selection():

    device = get_device("auto")

    assert device.type in {
        "cpu",
        "cuda",
    }


def test_checkpoint_roundtrip(
    tmp_path: Path,
):

    model = EDSR(
        scale=2,
        channels=16,
        num_blocks=2,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-4,
    )

    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=2,
        )
    )

    manager = CheckpointManager(
        tmp_path / "checkpoints"
    )

    path = manager.save(
        filename="test.pt",
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=1,
        best_metric=30.0,
        config={
            "test": True
        },
        metadata={
            "model": "test"
        },
    )

    assert path.exists()

    loaded_model = EDSR(
        scale=2,
        channels=16,
        num_blocks=2,
    )

    checkpoint = (
        CheckpointManager.load(
            path,
            loaded_model,
            device="cpu",
        )
    )

    assert checkpoint["epoch"] == 1
    assert checkpoint["best_metric"] == 30.0
