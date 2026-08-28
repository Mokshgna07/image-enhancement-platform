from __future__ import annotations

import random
from pathlib import Path

import matplotlib.pyplot as plt
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
).resolve().parents[1]

CONFIG_PATH = (
    PROJECT_ROOT
    / "ml"
    / "configs"
    / "degradation.yaml"
)

TRAIN_DIR = (
    PROJECT_ROOT
    / "data"
    / "div2k"
    / "raw"
    / "DIV2K_train_HR"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "div2k"
    / "processed"
)


def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    config = load_config(
        CONFIG_PATH
    )

    images = sorted(
        TRAIN_DIR.glob("*.png")
    )

    image_path = random.choice(
        images
    )

    with Image.open(
        image_path
    ) as image:
        hr = image.convert(
            "RGB"
        ).copy()

    hr = center_crop(
        hr,
        config["hr_patch_size"],
    )

    pipeline = (
        DegradationPipeline(
            config,
            seed=None,
        )
    )

    lr = pipeline(
        hr,
        (
            config["lr_patch_size"],
            config["lr_patch_size"],
        ),
    )

    output_path = (
        OUTPUT_DIR
        / "degradation_example.png"
    )

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(10, 5),
    )

    axes[0].imshow(hr)
    axes[0].set_title(
        "HR Target (192×192)"
    )
    axes[0].axis("off")

    axes[1].imshow(lr)
    axes[1].set_title(
        "Degraded LR (48×48)"
    )
    axes[1].axis("off")

    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=150,
    )

    plt.close(figure)

    print(
        f"Saved: {output_path}"
    )


if __name__ == "__main__":
    main()
