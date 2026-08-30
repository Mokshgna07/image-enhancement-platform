from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ml.src.super_resolution.checkpoints import (
    CheckpointManager,
)
from ml.src.super_resolution.config import (
    load_config,
)
from ml.src.super_resolution.datasets import (
    SuperResolutionDataset,
)
from ml.src.super_resolution.losses import (
    build_loss,
)
from ml.src.super_resolution.metrics import (
    psnr,
    ssim,
)
from ml.src.super_resolution.models import (
    EDSR,
)
from ml.src.super_resolution.models.model_utils import (
    count_parameters,
)
from ml.src.super_resolution.training import (
    EarlyStopping,
    JSONLLogger,
    get_device,
    set_seed,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_split(
    path: str | Path,
) -> list[Path]:

    path = Path(path)

    if not path.is_absolute():
        path = PROJECT_ROOT / path

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:

        return [
            PROJECT_ROOT / line.strip()
            for line in file
            if line.strip()
        ]


def build_dataloader(
    paths,
    config,
    split,
    batch_size,
    num_workers,
    shuffle,
):
    dataset = SuperResolutionDataset(
        image_paths=paths,
        config=config,
        split=split,
        seed=42,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=False,
        persistent_workers=(
            num_workers > 0
        ),
    )


def build_scheduler(
    optimizer,
    config,
    epochs,
):
    scheduler_config = config[
        "training"
    ]["scheduler"]

    scheduler_type = scheduler_config[
        "type"
    ].lower()

    if scheduler_type == "cosine":

        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=epochs,
            eta_min=float(
                scheduler_config[
                    "min_learning_rate"
                ]
            ),
        )

    raise ValueError(
        f"Unsupported scheduler: "
        f"{scheduler_type}"
    )


def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device,
    scaler,
    use_amp,
    max_grad_norm,
    epoch,
    log_every,
):
    model.train()

    running_loss = 0.0

    for batch_index, batch in enumerate(
        loader,
        start=1,
    ):

        lr = batch["lr"].to(
            device,
            non_blocking=True,
        )

        hr = batch["hr"].to(
            device,
            non_blocking=True,
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        with torch.amp.autocast(
            device_type="cuda",
            enabled=use_amp,
        ):

            sr = model(lr)

            loss = criterion(
                sr,
                hr,
            )

        if scaler is not None:

            scaler.scale(
                loss
            ).backward()

            scaler.unscale_(
                optimizer
            )

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_grad_norm,
            )

            scaler.step(
                optimizer
            )

            scaler.update()

        else:

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_grad_norm,
            )

            optimizer.step()

        running_loss += loss.item()

        if (
            batch_index % log_every == 0
            or batch_index == len(loader)
        ):

            average = (
                running_loss
                / batch_index
            )

            print(
                f"Epoch {epoch} | "
                f"Batch "
                f"{batch_index}/{len(loader)} | "
                f"Loss: {average:.6f}"
            )

    return (
        running_loss
        / len(loader)
    )


@torch.no_grad()
def validate(
    model,
    loader,
    criterion,
    device,
):
    model.eval()

    total_loss = 0.0
    total_psnr = 0.0
    total_ssim = 0.0

    batches = 0

    for batch in loader:

        lr = batch["lr"].to(
            device,
            non_blocking=True,
        )

        hr = batch["hr"].to(
            device,
            non_blocking=True,
        )

        sr = model(lr)

        sr = sr.clamp(
            0.0,
            1.0,
        )

        loss = criterion(
            sr,
            hr,
        )

        total_loss += loss.item()
        total_psnr += psnr(
            sr,
            hr,
        )
        total_ssim += ssim(
            sr,
            hr,
        )

        batches += 1

    return {
        "loss": total_loss / batches,
        "psnr": total_psnr / batches,
        "ssim": total_ssim / batches,
    }


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        required=True,
    )

    parser.add_argument(
        "--resume",
        default=None,
    )

    args = parser.parse_args()

    config = load_config(
        PROJECT_ROOT
        / args.config
        if not Path(
            args.config
        ).is_absolute()
        else args.config
    )

    seed = int(
        config["experiment"]["seed"]
    )

    set_seed(seed)

    device = get_device(
        config["training"]["device"]
    )

    print(
        f"Device: {device}"
    )

    if device.type == "cuda":

        print(
            f"CUDA device: "
            f"{torch.cuda.get_device_name(0)}"
        )

    experiment_dir = (
        PROJECT_ROOT
        / config["experiment"][
            "output_dir"
        ]
    )

    experiment_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    with (
        experiment_dir
        / "config.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            config,
            file,
            indent=2,
        )

    degradation_config = load_config(
        PROJECT_ROOT
        / config["data"]["config_path"]
    )

    train_paths = load_split(
        config["data"]["train_split"]
    )

    val_paths = load_split(
        config["data"]["val_split"]
    )

    train_loader = build_dataloader(
        train_paths,
        degradation_config,
        "train",
        config["data"]["batch_size"],
        config["data"]["num_workers"],
        True,
    )

    val_loader = build_dataloader(
        val_paths,
        degradation_config,
        "val",
        config["data"]["batch_size"],
        config["data"]["num_workers"],
        False,
    )

    model = EDSR(
        scale=config["model"]["scale"],
        channels=config["model"]["channels"],
        num_blocks=config["model"]["num_blocks"],
        residual_scale=config["model"][
            "residual_scale"
        ],
    ).to(device)

    parameter_count = count_parameters(
        model
    )

    print(
        f"Model parameters: "
        f"{parameter_count:,}"
    )

    criterion = build_loss(
        config["training"]["loss"]
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(
            config["training"][
                "optimizer"
            ]["learning_rate"]
        ),
        weight_decay=float(
            config["training"][
                "optimizer"
            ].get(
                "weight_decay",
                0.0,
            )
        ),
    )

    epochs = int(
        config["training"]["epochs"]
    )

    scheduler = build_scheduler(
        optimizer,
        config,
        epochs,
    )

    use_amp = (
        bool(
            config["training"][
                "mixed_precision"
            ]["enabled"]
        )
        and device.type == "cuda"
    )

    scaler = (
        torch.amp.GradScaler(
            "cuda",
            enabled=True,
        )
        if use_amp
        else None
    )

    checkpoint_manager = (
        CheckpointManager(
            experiment_dir
            / "checkpoints"
        )
    )

    logger = JSONLLogger(
        experiment_dir
        / "metrics.jsonl"
    )

    start_epoch = 1
    best_psnr = float("-inf")

    metadata = {
        "model": "EDSR-inspired",
        "parameters": parameter_count,
        "scale": config["model"][
            "scale"
        ],
        "device": str(device),
        "torch_version": torch.__version__,
    }

    if args.resume:

        checkpoint_path = (
            Path(args.resume)
        )

        if not checkpoint_path.is_absolute():
            checkpoint_path = (
                PROJECT_ROOT
                / checkpoint_path
            )

        print(
            f"Resuming from: "
            f"{checkpoint_path}"
        )

        checkpoint = (
            CheckpointManager.load(
                checkpoint_path,
                model,
                optimizer,
                scheduler,
                scaler,
                device,
            )
        )

        start_epoch = (
            int(
                checkpoint["epoch"]
            )
            + 1
        )

        best_psnr = float(
            checkpoint.get(
                "best_metric",
                float("-inf"),
            )
        )

    early_stopping = None

    early_config = config[
        "training"
    ]["early_stopping"]

    if early_config["enabled"]:

        early_stopping = EarlyStopping(
            patience=int(
                early_config[
                    "patience"
                ]
            ),
            min_delta=float(
                early_config[
                    "min_delta"
                ]
            ),
        )

        if best_psnr != float(
            "-inf"
        ):
            early_stopping.best = (
                best_psnr
            )

    for epoch in range(
        start_epoch,
        epochs + 1,
    ):

        print()
        print(
            f"========== Epoch "
            f"{epoch}/{epochs} =========="
        )

        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            scaler=scaler,
            use_amp=use_amp,
            max_grad_norm=float(
                config["training"][
                    "gradient"
                ]["max_norm"]
            ),
            epoch=epoch,
            log_every=int(
                config["training"][
                    "log_every"
                ]
            ),
        )

        validation = validate(
            model,
            val_loader,
            criterion,
            device,
        )

        scheduler.step()

        learning_rate = (
            optimizer.param_groups[0][
                "lr"
            ]
        )

        print(
            f"Epoch {epoch} complete | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: "
            f"{validation['loss']:.6f} | "
            f"PSNR: "
            f"{validation['psnr']:.4f} | "
            f"SSIM: "
            f"{validation['ssim']:.4f} | "
            f"LR: "
            f"{learning_rate:.8f}"
        )

        logger.log(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": validation[
                    "loss"
                ],
                "psnr": validation[
                    "psnr"
                ],
                "ssim": validation[
                    "ssim"
                ],
                "learning_rate": learning_rate,
            }
        )

        checkpoint_manager.save(
            filename="latest.pt",
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            best_metric=best_psnr,
            config=config,
            metadata=metadata,
            scaler=scaler,
        )

        if (
            validation["psnr"]
            > best_psnr
        ):

            best_psnr = validation[
                "psnr"
            ]

            checkpoint_manager.save(
                filename="best.pt",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                best_metric=best_psnr,
                config=config,
                metadata=metadata,
                scaler=scaler,
            )

            print(
                f"New best model: "
                f"PSNR={best_psnr:.4f}"
            )

        if (
            early_stopping is not None
            and early_stopping.step(
                validation["psnr"]
            )
        ):

            print(
                "Early stopping triggered."
            )

            break

    print()
    print(
        "Training completed."
    )


if __name__ == "__main__":
    main()
