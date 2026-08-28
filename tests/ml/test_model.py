from pathlib import Path

import torch

from ml.src.super_resolution.models import (
    EDSR,
)
from ml.src.super_resolution.models.model_utils import (
    count_parameters,
    load_model,
    save_model,
)


def test_model_forward_and_save_load(
    tmp_path: Path,
):
    model = EDSR(
        scale=4,
        channels=32,
        num_blocks=4,
    )

    model.eval()

    batch_size = 2
    height = 48
    width = 48

    dummy_lr = torch.randn(
        batch_size,
        3,
        height,
        width,
    )

    with torch.no_grad():
        output = model(
            dummy_lr
        )

    assert output.shape == (
        batch_size,
        3,
        192,
        192,
    )

    assert count_parameters(
        model
    ) > 0

    model_path = (
        tmp_path
        / "edsr_test.pt"
    )

    save_model(
        model,
        model_path,
    )

    assert model_path.exists()

    loaded_model = EDSR(
        scale=4,
        channels=32,
        num_blocks=4,
    )

    load_model(
        loaded_model,
        model_path,
    )

    loaded_model.eval()

    with torch.no_grad():
        loaded_output = (
            loaded_model(
                dummy_lr
            )
        )

    assert torch.allclose(
        output,
        loaded_output,
    )
def test_model_supports_2x():
    model = EDSR(
        scale=2,
        channels=32,
        num_blocks=4,
    )

    model.eval()

    dummy_lr = torch.randn(
        1,
        3,
        48,
        48,
    )

    with torch.no_grad():
        output = model(
            dummy_lr
        )

    assert output.shape == (
        1,
        3,
        96,
        96,
    )
