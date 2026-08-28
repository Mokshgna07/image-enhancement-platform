from __future__ import annotations

from pathlib import Path
from typing import Literal

from PIL import Image
from torch.utils.data import Dataset

from .degradation import DegradationPipeline
from .image_utils import pil_to_tensor
from .transforms import (
    center_crop,
    random_crop,
    random_flip_rotate,
)


class SuperResolutionDataset(
    Dataset
):

    def __init__(
        self,
        image_paths: list[str | Path],
        config: dict,
        split: Literal[
            "train",
            "val",
            "test",
        ],
        seed: int = 42,
    ):

        self.image_paths = [
            Path(path)
            for path in image_paths
        ]

        self.config = config
        self.split = split

        self.scale_factor = int(
            config["scale_factor"]
        )

        self.hr_patch_size = int(
            config["hr_patch_size"]
        )

        self.lr_patch_size = (
            self.hr_patch_size
            // self.scale_factor
        )

        if (
            self.hr_patch_size
            % self.scale_factor
            != 0
        ):
            raise ValueError(
                "HR patch size must be "
                "divisible by scale factor."
            )

        self.degradation = (
            DegradationPipeline(
                config,
                seed=seed,
            )
        )

    def __len__(self) -> int:
        return len(
            self.image_paths
        )

    def _load_image(
        self,
        path: Path,
    ) -> Image.Image:

        with Image.open(path) as image:
            return image.convert(
                "RGB"
            ).copy()

    def __getitem__(
        self,
        index: int,
    ):

        path = self.image_paths[index]

        hr = self._load_image(path)

        if self.split == "train":

            hr = random_crop(
                hr,
                self.hr_patch_size,
            )

            if self.config[
                "train"
            ].get(
                "random_flip",
                True,
            ) or self.config[
                "train"
            ].get(
                "random_rotation",
                True,
            ):
                hr = random_flip_rotate(
                    hr
                )

        else:

            hr = center_crop(
                hr,
                self.hr_patch_size,
            )

        lr = self.degradation(
            hr,
            (
                self.lr_patch_size,
                self.lr_patch_size,
            ),
        )

        hr_tensor = pil_to_tensor(
            hr
        )

        lr_tensor = pil_to_tensor(
            lr
        )

        return {
            "lr": lr_tensor,
            "hr": hr_tensor,
            "path": str(path),
        }
