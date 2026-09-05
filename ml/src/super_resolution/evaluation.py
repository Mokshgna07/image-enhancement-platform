
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from skimage.metrics import structural_similarity


def calculate_psnr(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> float:
    prediction = prediction.detach().cpu().clamp(0, 1)
    target = target.detach().cpu().clamp(0, 1)

    mse = torch.mean((prediction - target) ** 2).item()

    if mse == 0:
        return float("inf")

    return float(10.0 * np.log10(1.0 / mse))


def calculate_single_ssim(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> float:
    prediction = prediction.detach().cpu().clamp(0, 1)
    target = target.detach().cpu().clamp(0, 1)

    prediction = prediction[0].permute(1, 2, 0).numpy()
    target = target[0].permute(1, 2, 0).numpy()

    return float(
        structural_similarity(
            prediction,
            target,
            channel_axis=2,
            data_range=1.0,
        )
    )


def evaluate_batch(
    model,
    lr: torch.Tensor,
    hr: torch.Tensor,
    device: torch.device,
):
    model.eval()

    lr = lr.to(device)
    hr = hr.to(device)

    with torch.inference_mode():
        sr = model(lr)

    return {
        "sr": sr,
        "psnr": calculate_psnr(sr, hr),
        "ssim": calculate_single_ssim(sr, hr),
    }


def summarize_metrics(records):
    if not records:
        raise ValueError("No evaluation records provided.")

    def stats(key):
        values = np.array(
            [record[key] for record in records],
            dtype=np.float64,
        )

        return {
            "mean": float(values.mean()),
            "std": float(values.std()),
            "min": float(values.min()),
            "max": float(values.max()),
        }

    return {
        "num_images": len(records),
        "psnr": stats("psnr"),
        "ssim": stats("ssim"),
    }


def benchmark_model(
    model,
    input_tensor: torch.Tensor,
    device: torch.device,
    warmup: int = 10,
    iterations: int = 100,
):
    model.eval()
    input_tensor = input_tensor.to(device)

    with torch.inference_mode():
        for _ in range(warmup):
            model(input_tensor)

    if device.type == "cuda":
        torch.cuda.synchronize()

    import time

    timings = []

    with torch.inference_mode():
        for _ in range(iterations):
            if device.type == "cuda":
                torch.cuda.synchronize()

            start = time.perf_counter()

            model(input_tensor)

            if device.type == "cuda":
                torch.cuda.synchronize()

            end = time.perf_counter()

            timings.append(
                (end - start) * 1000.0
            )

    timings.sort()

    mean_ms = sum(timings) / len(timings)
    median_ms = timings[len(timings) // 2]

    return {
        "mean_latency_ms": mean_ms,
        "median_latency_ms": median_ms,
        "throughput_images_per_second": 1000.0 / mean_ms,
    }


def tensor_to_image(tensor: torch.Tensor) -> Image.Image:
    tensor = tensor.detach().cpu().clamp(0, 1)

    if tensor.ndim == 4:
        tensor = tensor[0]

    array = tensor.permute(1, 2, 0).numpy()
    array = (array * 255.0).round().astype(np.uint8)

    return Image.fromarray(array)


def save_json(data, path: str | Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as file:
        json.dump(data, file, indent=2)
