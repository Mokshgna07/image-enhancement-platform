from pathlib import Path

import pytest
from PIL import Image

from super_resolution.inference import (
    ImageValidationError,
    ModelConfigurationError,
    SuperResolutionService,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "ml/configs/training.yaml"
CHECKPOINT_4X = (
    PROJECT_ROOT
    / "runs/edsr_x4_baseline/checkpoints/best.pt"
)


@pytest.fixture
def service():
    return SuperResolutionService(
        config_path=CONFIG_PATH,
        checkpoint_paths={
            4: CHECKPOINT_4X,
        },
        device="cpu",
        model_version="edsr_x4_baseline",
    )


def test_real_4x_inference(service, tmp_path):
    input_path = tmp_path / "sample.png"
    output_path = tmp_path / "enhanced.png"

    image = Image.new(
        "RGB",
        (48, 48),
    )
    image.save(input_path)

    result = service.enhance(
        image_path=input_path,
        output_path=output_path,
        scale=4,
    )

    assert output_path.exists()

    assert result.input_width == 48
    assert result.input_height == 48

    assert result.output_width == 192
    assert result.output_height == 192

    assert result.input_format == "PNG"
    assert result.output_format == "PNG"

    assert result.processing_time_ms > 0
    assert result.model_version == "edsr_x4_baseline"
    assert result.scale == 4
    assert result.device == "cpu"

    with Image.open(output_path) as enhanced:
        assert enhanced.size == (192, 192)
        assert enhanced.mode == "RGB"


def test_missing_image_is_rejected(service, tmp_path):
    with pytest.raises(ImageValidationError):
        service.enhance(
            image_path=tmp_path / "missing.png",
            output_path=tmp_path / "output.png",
            scale=4,
        )


def test_unsupported_scale_is_rejected(service, tmp_path):
    input_path = tmp_path / "sample.png"

    Image.new(
        "RGB",
        (48, 48),
    ).save(input_path)

    with pytest.raises(Exception):
        service.enhance(
            image_path=input_path,
            output_path=tmp_path / "output.png",
            scale=8,
        )


def test_missing_2x_checkpoint_is_reported(service, tmp_path):
    input_path = tmp_path / "sample.png"

    Image.new(
        "RGB",
        (48, 48),
    ).save(input_path)

    with pytest.raises(ModelConfigurationError):
        service.enhance(
            image_path=input_path,
            output_path=tmp_path / "output.png",
            scale=2,
        )
