from __future__ import annotations

from io import BytesIO

import numpy as np
from PIL import Image

import torch


def pil_to_numpy(image: Image.Image) -> np.ndarray:
    return np.asarray(
        image.convert("RGB")
    ).copy()


def numpy_to_pil(
    array: np.ndarray,
) -> Image.Image:
    array = np.clip(
        array,
        0,
        255,
    ).astype(np.uint8)

    return Image.fromarray(
        array,
        mode="RGB",
    )


def pil_to_tensor(
    image: Image.Image,
) -> torch.Tensor:
    array = np.asarray(
        image.convert("RGB"),
        dtype=np.float32,
    ) / 255.0

    array = np.transpose(
        array,
        (2, 0, 1),
    )

    return torch.from_numpy(
        array.copy()
    ).float()


def jpeg_roundtrip(
    image: Image.Image,
    quality: int,
) -> Image.Image:

    buffer = BytesIO()

    image.save(
        buffer,
        format="JPEG",
        quality=int(quality),
    )

    buffer.seek(0)

    compressed = Image.open(
        buffer
    ).convert("RGB")

    return compressed.copy()
