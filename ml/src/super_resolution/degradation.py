from __future__ import annotations

import random

import cv2
import numpy as np
from PIL import Image, ImageEnhance

from .image_utils import jpeg_roundtrip


INTERPOLATION_MODES = {
    "nearest": cv2.INTER_NEAREST,
    "bilinear": cv2.INTER_LINEAR,
    "bicubic": cv2.INTER_CUBIC,
    "area": cv2.INTER_AREA,
    "lanczos": cv2.INTER_LANCZOS4,
}


class DegradationPipeline:

    def __init__(
        self,
        config: dict,
        seed: int | None = None,
    ):
        self.config = config
        self.degradation_config = (
            config["degradation"]
        )

        self.seed = seed

        if seed is not None:
            self.rng = random.Random(seed)
            self.numpy_rng = np.random.default_rng(
                seed
            )
        else:
            self.rng = random.Random()
            self.numpy_rng = np.random.default_rng()

    def _should_apply(
        self,
        config: dict,
    ) -> bool:

        if not config.get(
            "enabled",
            True,
        ):
            return False

        probability = float(
            config.get(
                "probability",
                1.0,
            )
        )

        return (
            self.rng.random()
            < probability
        )

    def apply_blur(
        self,
        image: Image.Image,
    ) -> Image.Image:

        config = self.degradation_config["blur"]

        if not self._should_apply(config):
            return image

        min_kernel, max_kernel = (
            config["kernel_size_range"]
        )

        possible_kernels = list(
            range(
                min_kernel,
                max_kernel + 1,
                2,
            )
        )

        kernel_size = self.rng.choice(
            possible_kernels
        )

        sigma_min, sigma_max = (
            config["sigma_range"]
        )

        sigma = self.rng.uniform(
            sigma_min,
            sigma_max,
        )

        array = np.asarray(
            image.convert("RGB")
        )

        blurred = cv2.GaussianBlur(
            array,
            (
                kernel_size,
                kernel_size,
            ),
            sigmaX=sigma,
            sigmaY=sigma,
        )

        return Image.fromarray(
            blurred
        )

    def resize_to_lr(
        self,
        image: Image.Image,
        target_size: tuple[int, int],
    ) -> Image.Image:

        config = self.degradation_config["resize"]

        if not self._should_apply(config):
            return image.resize(
                target_size,
                Image.Resampling.BICUBIC,
            )

        interpolation_name = (
            self.rng.choice(
                config["interpolation"]
            )
        )

        interpolation = (
            INTERPOLATION_MODES[
                interpolation_name
            ]
        )

        array = np.asarray(
            image.convert("RGB")
        )

        resized = cv2.resize(
            array,
            target_size,
            interpolation=interpolation,
        )

        return Image.fromarray(
            resized
        )

    def add_gaussian_noise(
        self,
        image: Image.Image,
    ) -> Image.Image:

        config = (
            self.degradation_config[
                "gaussian_noise"
            ]
        )

        if not self._should_apply(config):
            return image

        sigma_min, sigma_max = (
            config["sigma_range"]
        )

        sigma = self.rng.uniform(
            sigma_min,
            sigma_max,
        )

        array = np.asarray(
            image.convert("RGB"),
            dtype=np.float32,
        ) / 255.0

        noise = self.numpy_rng.normal(
            0.0,
            sigma,
            array.shape,
        ).astype(np.float32)

        noisy = np.clip(
            array + noise,
            0.0,
            1.0,
        )

        return Image.fromarray(
            (
                noisy * 255.0
            ).round().astype(
                np.uint8
            )
        )

    def add_poisson_noise(
        self,
        image: Image.Image,
    ) -> Image.Image:

        config = (
            self.degradation_config[
                "poisson_noise"
            ]
        )

        if not self._should_apply(config):
            return image

        scale_min, scale_max = (
            config["scale_range"]
        )

        scale = self.rng.uniform(
            scale_min,
            scale_max,
        )

        array = np.asarray(
            image.convert("RGB"),
            dtype=np.float32,
        ) / 255.0

        noisy = (
            self.numpy_rng.poisson(
                np.clip(
                    array,
                    0.0,
                    1.0,
                ) * scale
            )
            / scale
        )

        noisy = np.clip(
            noisy,
            0.0,
            1.0,
        )

        return Image.fromarray(
            (
                noisy * 255.0
            ).round().astype(
                np.uint8
            )
        )

    def apply_jpeg(
        self,
        image: Image.Image,
    ) -> Image.Image:

        config = (
            self.degradation_config[
                "jpeg"
            ]
        )

        if not self._should_apply(config):
            return image

        quality_min, quality_max = (
            config["quality_range"]
        )

        quality = self.rng.randint(
            quality_min,
            quality_max,
        )

        return jpeg_roundtrip(
            image,
            quality,
        )

    def apply_color_jitter(
        self,
        image: Image.Image,
    ) -> Image.Image:

        config = (
            self.degradation_config[
                "color_jitter"
            ]
        )

        if not self._should_apply(config):
            return image

        brightness = self.rng.uniform(
            1.0 - config["brightness"],
            1.0 + config["brightness"],
        )

        contrast = self.rng.uniform(
            1.0 - config["contrast"],
            1.0 + config["contrast"],
        )

        saturation = self.rng.uniform(
            1.0 - config["saturation"],
            1.0 + config["saturation"],
        )

        image = ImageEnhance.Brightness(
            image
        ).enhance(brightness)

        image = ImageEnhance.Contrast(
            image
        ).enhance(contrast)

        image = ImageEnhance.Color(
            image
        ).enhance(saturation)

        return image

    def apply_grayscale(
        self,
        image: Image.Image,
    ) -> Image.Image:

        config = (
            self.degradation_config[
                "grayscale"
            ]
        )

        if not self._should_apply(config):
            return image

        return image.convert(
            "L"
        ).convert("RGB")

    def __call__(
        self,
        image: Image.Image,
        target_size: tuple[int, int],
    ) -> Image.Image:

        image = image.convert("RGB")

        # 1. Optical-style blur
        image = self.apply_blur(image)

        # 2. Color/exposure variation
        image = self.apply_color_jitter(image)

        # 3. Optional grayscale degradation
        image = self.apply_grayscale(image)

        # 4. Resolution reduction
        image = self.resize_to_lr(
            image,
            target_size,
        )

        # 5. Sensor-like noise
        image = self.add_gaussian_noise(
            image
        )

        # 6. Shot-noise approximation
        image = self.add_poisson_noise(
            image
        )

        # 7. Compression artifacts
        image = self.apply_jpeg(
            image
        )

        return image.convert("RGB")
