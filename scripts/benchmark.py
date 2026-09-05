
from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import yaml

from super_resolution.model import EDSR, count_parameters


def load_config(path: str):
    with open(path, "r") as file:
        return yaml.safe_load(file)


def load_model(config, checkpoint_path, device):
    model_config = config["model"]

    model = EDSR(
        scale=int(model_config["scale"]),
        channels=int(model_config["channels"]),
        num_blocks=int(model_config["num_blocks"]),
        residual_scale=float(model_config["residual_scale"]),
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model


def benchmark(model, device, warmup, iterations):
    x = torch.randn(
        1,
        3,
        48,
        48,
        device=device,
    )

    with torch.inference_mode():
        for _ in range(warmup):
            model(x)

    if device.type == "cuda":
        torch.cuda.synchronize()

    timings = []

    with torch.inference_mode():
        for _ in range(iterations):
            if device.type == "cuda":
                torch.cuda.synchronize()

            start = time.perf_counter()

            model(x)

            if device.type == "cuda":
                torch.cuda.synchronize()

            end = time.perf_counter()

            timings.append(
                (end - start) * 1000.0
            )

    timings.sort()

    mean_ms = sum(timings) / len(timings)
    median_ms = timings[len(timings) // 2]
    throughput = 1000.0 / mean_ms

    return {
        "input_shape": list(x.shape),
        "output_shape": [1, 3, 192, 192],
        "warmup_iterations": warmup,
        "timed_iterations": iterations,
        "mean_latency_ms": mean_ms,
        "median_latency_ms": median_ms,
        "throughput_images_per_second": throughput,
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        default="ml/configs/training.yaml",
    )

    parser.add_argument(
        "--checkpoint",
        default="runs/edsr_x4_baseline/checkpoints/best.pt",
    )

    parser.add_argument(
        "--warmup",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--iterations",
        type=int,
        default=100,
    )

    args = parser.parse_args()

    config = load_config(args.config)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)

    if device.type == "cuda":
        print(
            "GPU:",
            torch.cuda.get_device_name(0),
        )

    model = load_model(
        config,
        args.checkpoint,
        device,
    )

    print(
        "Parameters:",
        count_parameters(model),
    )

    results = benchmark(
        model=model,
        device=device,
        warmup=args.warmup,
        iterations=args.iterations,
    )

    print("\n===== EDSR x4 Benchmark =====")

    for key, value in results.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
