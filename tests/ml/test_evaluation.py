
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, "ml/src")

from super_resolution.evaluation import (
    calculate_psnr,
    calculate_single_ssim,
)
from super_resolution.model import EDSR


def test_psnr_identical_images():
    image = torch.rand(1, 3, 32, 32)

    score = calculate_psnr(image, image)

    assert score == float("inf")


def test_psnr_positive_for_different_images():
    prediction = torch.zeros(1, 3, 32, 32)
    target = torch.ones(1, 3, 32, 32)

    score = calculate_psnr(prediction, target)

    assert score >= 0.0


def test_ssim_range():
    image = torch.rand(1, 3, 32, 32)

    score = calculate_single_ssim(image, image)

    assert 0.99 <= score <= 1.0


def test_model_output_shape():
    model = EDSR(
        scale=4,
        channels=64,
        num_blocks=16,
        residual_scale=0.1,
    )

    model.eval()

    lr = torch.rand(1, 3, 48, 48)

    with torch.no_grad():
        sr = model(lr)

    assert sr.shape == (1, 3, 192, 192)


def test_model_parameter_count():
    model = EDSR(
        scale=4,
        channels=64,
        num_blocks=16,
        residual_scale=0.1,
    )

    parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    assert parameters == 1_517_571
