from __future__ import annotations

import math

import torch
from torch import Tensor


def psnr(
    prediction: Tensor,
    target: Tensor,
    max_value: float = 1.0,
) -> float:
    """
    Calculate PSNR for tensors normalized to [0, 1].
    """

    prediction = prediction.detach().float()
    target = target.detach().float()

    mse = torch.mean(
        (prediction - target).pow(2)
    ).item()

    if mse == 0:
        return float("inf")

    return 10.0 * math.log10(
        (max_value ** 2) / mse
    )


def ssim(
    prediction: Tensor,
    target: Tensor,
) -> float:
    """
    Calculate batch-averaged SSIM using
    scikit-image.
    """

    from skimage.metrics import structural_similarity

    prediction = (
        prediction.detach()
        .float()
        .cpu()
        .clamp(0.0, 1.0)
    )

    target = (
        target.detach()
        .float()
        .cpu()
        .clamp(0.0, 1.0)
    )

    scores = []

    for pred, true in zip(
        prediction,
        target,
    ):
        pred_np = (
            pred.permute(
                1,
                2,
                0,
            ).numpy()
        )

        true_np = (
            true.permute(
                1,
                2,
                0,
            ).numpy()
        )

        score = structural_similarity(
            true_np,
            pred_np,
            data_range=1.0,
            channel_axis=2,
        )

        scores.append(score)

    return sum(scores) / len(scores)
