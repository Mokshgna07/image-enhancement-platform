from pathlib import Path

from torch.utils.data import DataLoader

from ml.src.super_resolution.config import (
    load_config,
)
from ml.src.super_resolution.datasets import (
    SuperResolutionDataset,
)


PROJECT_ROOT = Path(
    __file__
).resolve().parents[2]


def load_split(
    split: str,
):
    split_file = (
        PROJECT_ROOT
        / "data"
        / "div2k"
        / "splits"
        / f"{split}.txt"
    )

    with split_file.open(
        "r",
        encoding="utf-8",
    ) as file:
        return [
            PROJECT_ROOT / line.strip()
            for line in file
            if line.strip()
        ]


def test_dataset():

    config = load_config(
        PROJECT_ROOT
        / "ml"
        / "configs"
        / "degradation.yaml"
    )

    train_paths = load_split(
        "train"
    )

    dataset = (
        SuperResolutionDataset(
            train_paths,
            config,
            split="train",
            seed=42,
        )
    )

    assert len(dataset) == 700

    sample = dataset[0]

    assert sample["lr"].shape == (
        3,
        48,
        48,
    )

    assert sample["hr"].shape == (
        3,
        192,
        192,
    )

    assert (
        sample["lr"].min()
        >= 0
    )

    assert (
        sample["lr"].max()
        <= 1
    )

    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=0,
    )

    batch = next(
        iter(loader)
    )

    assert batch["lr"].shape == (
        4,
        3,
        48,
        48,
    )

    assert batch["hr"].shape == (
        4,
        3,
        192,
        192,
    )
def test_validation_degradation_is_deterministic():
    from ml.src.super_resolution.datasets import (
        SuperResolutionDataset,
    )
    from ml.src.super_resolution.config import (
        load_config,
    )
    import torch

    config = load_config(
        "ml/configs/degradation.yaml"
    )

    image_paths = [
        "data/div2k/raw/DIV2K_train_HR/0001.png"
    ]

    dataset = SuperResolutionDataset(
        image_paths=image_paths,
        config=config,
        split="val",
        seed=42,
    )

    sample_a = dataset[0]
    sample_b = dataset[0]

    assert torch.equal(
        sample_a["lr"],
        sample_b["lr"],
    )

    assert torch.equal(
        sample_a["hr"],
        sample_b["hr"],
    )
