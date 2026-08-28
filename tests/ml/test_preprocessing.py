from pathlib import Path

from PIL import Image

from ml.src.super_resolution.config import (
    load_config,
)
from ml.src.super_resolution.degradation import (
    DegradationPipeline,
)
from ml.src.super_resolution.transforms import (
    center_crop,
)


PROJECT_ROOT = Path(
    __file__
).resolve().parents[2]


def test_hr_to_lr():

    config = load_config(
        PROJECT_ROOT
        / "ml"
        / "configs"
        / "degradation.yaml"
    )

    image_paths = sorted(
        (
            PROJECT_ROOT
            / "data"
            / "div2k"
            / "raw"
            / "DIV2K_train_HR"
        ).glob("*.png")
    )

    assert len(image_paths) == 800

    with Image.open(
        image_paths[0]
    ) as image:
        hr = image.convert(
            "RGB"
        ).copy()

    hr = center_crop(
        hr,
        config["hr_patch_size"],
    )

    pipeline = DegradationPipeline(
        config,
        seed=42,
    )

    lr = pipeline(
        hr,
        (
            config["lr_patch_size"],
            config["lr_patch_size"],
        ),
    )

    assert hr.size == (
        192,
        192,
    )

    assert lr.size == (
        48,
        48,
    )

    assert hr.mode == "RGB"
    assert lr.mode == "RGB"
