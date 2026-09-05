
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image
from skimage.metrics import structural_similarity
from torch.utils.data import DataLoader

import lpips

from super_resolution.datasets import SuperResolutionDataset
from super_resolution.model import EDSR


def calculate_psnr(prediction: torch.Tensor, target: torch.Tensor) -> float:
    prediction = prediction.detach().cpu().clamp(0, 1)
    target = target.detach().cpu().clamp(0, 1)

    mse = torch.mean((prediction - target) ** 2).item()

    if mse == 0:
        return float("inf")

    return 10.0 * np.log10(1.0 / mse)


def calculate_ssim(prediction: torch.Tensor, target: torch.Tensor) -> float:
    prediction = prediction.detach().cpu().clamp(0, 1)
    target = target.detach().cpu().clamp(0, 1)

    pred = prediction[0].permute(1, 2, 0).numpy()
    truth = target[0].permute(1, 2, 0).numpy()

    return float(
        structural_similarity(
            pred,
            truth,
            channel_axis=2,
            data_range=1.0,
        )
    )


def tensor_to_image(tensor: torch.Tensor) -> Image.Image:
    tensor = tensor.detach().cpu().clamp(0, 1)

    array = tensor[0].permute(1, 2, 0).numpy()
    array = (array * 255.0).round().astype(np.uint8)

    return Image.fromarray(array)


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


def evaluate(
    model,
    dataset,
    device,
    output_dir,
    lpips_model,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    records = []

    with torch.no_grad():
        for index in range(len(dataset)):
            sample = dataset[index]

            lr = sample["lr"].unsqueeze(0).to(device)
            hr = sample["hr"].unsqueeze(0).to(device)

            start = time.perf_counter()

            sr = model(lr)

            if device.type == "cuda":
                torch.cuda.synchronize()

            inference_ms = (time.perf_counter() - start) * 1000.0

            psnr = calculate_psnr(sr, hr)
            ssim = calculate_ssim(sr, hr)

            # LPIPS expects RGB tensors in [-1, 1].
            lpips_value = float(
                lpips_model(
                    sr * 2.0 - 1.0,
                    hr * 2.0 - 1.0,
                ).mean().item()
            )

            image_id = Path(sample["path"]).stem

            image_dir = output_dir / image_id
            image_dir.mkdir(parents=True, exist_ok=True)

            tensor_to_image(lr).save(image_dir / "lr.png")
            tensor_to_image(sr).save(image_dir / "sr.png")
            tensor_to_image(hr).save(image_dir / "hr.png")

            records.append(
                {
                    "image_id": image_id,
                    "psnr": psnr,
                    "ssim": ssim,
                    "lpips": lpips_value,
                    "inference_ms": inference_ms,
                }
            )

            if (index + 1) % 10 == 0:
                print(f"Evaluated {index + 1}/{len(dataset)}")

    return records


def summarize(records):
    def stats(key):
        values = np.array([record[key] for record in records], dtype=np.float64)

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
        "lpips": stats("lpips"),
        "inference_ms": stats("inference_ms"),
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
        "--split",
        choices=["val", "test"],
        default="test",
    )

    args = parser.parse_args()

    config = load_config(args.config)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)

    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    model = load_model(
        config,
        args.checkpoint,
        device,
    )

    degradation_config = load_config(
        config["data"]["config_path"]
    )

    split_path = config["data"][f"{args.split}_split"]

    with open(split_path, "r") as file:
        image_paths = [
            line.strip()
            for line in file
            if line.strip()
        ]

    dataset = SuperResolutionDataset(
        image_paths=image_paths,
        config=degradation_config,
        split=args.split,
        seed=int(config["experiment"]["seed"]),
    )

    print("Split:", args.split)
    print("Images:", len(dataset))

    lpips_model = lpips.LPIPS(net="alex").to(device)
    lpips_model.eval()

    output_dir = (
        Path("results")
        / "edsr_x4_baseline"
        / args.split
    )

    records = evaluate(
        model=model,
        dataset=dataset,
        device=device,
        output_dir=output_dir / "images",
        lpips_model=lpips_model,
    )

    summary = summarize(records)

    summary["model"] = {
        "parameters": sum(
            parameter.numel()
            for parameter in model.parameters()
        ),
        "checkpoint_mb": (
            Path(args.checkpoint).stat().st_size
            / (1024 ** 2)
        ),
    }

    with open(output_dir / "per_image.jsonl", "w") as file:
        for record in records:
            file.write(json.dumps(record) + "\n")

    with open(output_dir / "metrics.json", "w") as file:
        json.dump(summary, file, indent=2)

    print("\nEvaluation complete.")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
