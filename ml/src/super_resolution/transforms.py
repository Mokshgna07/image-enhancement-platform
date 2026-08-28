from __future__ import annotations

import random

from PIL import Image


def resize_if_needed(
    image: Image.Image,
    minimum_size: int,
) -> Image.Image:

    width, height = image.size

    if (
        width >= minimum_size
        and height >= minimum_size
    ):
        return image

    scale = max(
        minimum_size / width,
        minimum_size / height,
    )

    new_width = int(round(width * scale))
    new_height = int(round(height * scale))

    return image.resize(
        (new_width, new_height),
        Image.Resampling.BICUBIC,
    )


def random_crop(
    image: Image.Image,
    size: int,
) -> Image.Image:

    image = resize_if_needed(
        image,
        size,
    )

    width, height = image.size

    left = random.randint(
        0,
        width - size,
    )

    top = random.randint(
        0,
        height - size,
    )

    return image.crop(
        (
            left,
            top,
            left + size,
            top + size,
        )
    )


def center_crop(
    image: Image.Image,
    size: int,
) -> Image.Image:

    image = resize_if_needed(
        image,
        size,
    )

    width, height = image.size

    left = (width - size) // 2
    top = (height - size) // 2

    return image.crop(
        (
            left,
            top,
            left + size,
            top + size,
        )
    )


def random_flip_rotate(
    image: Image.Image,
) -> Image.Image:

    if random.random() < 0.5:
        image = image.transpose(
            Image.Transpose.FLIP_LEFT_RIGHT
        )

    if random.random() < 0.5:
        image = image.transpose(
            Image.Transpose.FLIP_TOP_BOTTOM
        )

    if random.random() < 0.5:
        image = image.transpose(
            random.choice(
                [
                    Image.Transpose.ROTATE_90,
                    Image.Transpose.ROTATE_180,
                    Image.Transpose.ROTATE_270,
                ]
            )
        )

    return image
