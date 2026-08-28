from __future__ import annotations

import random
from pathlib import Path


SEED = 42

PROJECT_ROOT = Path(__file__).resolve().parents[1]

TRAIN_HR_DIR = (
    PROJECT_ROOT
    / "data"
    / "div2k"
    / "raw"
    / "DIV2K_train_HR"
)

VALID_HR_DIR = (
    PROJECT_ROOT
    / "data"
    / "div2k"
    / "raw"
    / "DIV2K_valid_HR"
)

SPLIT_DIR = (
    PROJECT_ROOT
    / "data"
    / "div2k"
    / "splits"
)

IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
}


def get_images(directory: Path) -> list[Path]:
    if not directory.exists():
        raise FileNotFoundError(
            f"Dataset directory not found: {directory}"
        )

    return sorted(
        path
        for path in directory.rglob("*")
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def write_split(
    name: str,
    paths: list[Path],
) -> None:
    output_path = SPLIT_DIR / f"{name}.txt"

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        for path in paths:
            relative_path = path.relative_to(
                PROJECT_ROOT
            )

            file.write(
                str(relative_path)
                + "\n"
            )

    print(
        f"{name}: {len(paths)} images"
    )


def main() -> None:
    SPLIT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    train_images = get_images(
        TRAIN_HR_DIR
    )

    validation_images = get_images(
        VALID_HR_DIR
    )

    print(
        f"Found {len(train_images)} "
        "training HR images."
    )

    print(
        f"Found {len(validation_images)} "
        "validation HR images."
    )

    if len(train_images) != 800:
        raise RuntimeError(
            "Expected exactly 800 DIV2K "
            f"training images, found "
            f"{len(train_images)}."
        )

    if len(validation_images) != 100:
        raise RuntimeError(
            "Expected exactly 100 DIV2K "
            f"validation images, found "
            f"{len(validation_images)}."
        )

    rng = random.Random(SEED)

    shuffled_train = train_images.copy()

    rng.shuffle(shuffled_train)

    train_split = shuffled_train[:700]
    test_split = shuffled_train[700:]

    write_split(
        "train",
        train_split,
    )

    write_split(
        "val",
        validation_images,
    )

    write_split(
        "test",
        test_split,
    )

    print()
    print("Split creation completed.")
    print(f"Seed: {SEED}")


if __name__ == "__main__":
    main()
