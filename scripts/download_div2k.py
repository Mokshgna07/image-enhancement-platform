from __future__ import annotations

import shutil
import urllib.request
import zipfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = (
    PROJECT_ROOT
    / "data"
    / "div2k"
    / "raw"
)

BASE_URL = (
    "https://data.vision.ee.ethz.ch/cvl/DIV2K/"
)


FILES = {
    "DIV2K_train_HR.zip": "DIV2K_train_HR",
    "DIV2K_valid_HR.zip": "DIV2K_valid_HR",
}


def download(
    url: str,
    destination: Path,
):
    print(f"Downloading: {url}")

    urllib.request.urlretrieve(
        url,
        destination,
    )


def extract(
    archive: Path,
):
    print(f"Extracting: {archive}")

    with zipfile.ZipFile(
        archive,
        "r",
    ) as zip_file:
        zip_file.extractall(RAW_DIR)


def main():
    RAW_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    for filename, expected_dir in FILES.items():
        archive = RAW_DIR / filename
        extracted = RAW_DIR / expected_dir

        if extracted.exists():
            print(
                f"Already exists: {extracted}"
            )
            continue

        url = BASE_URL + filename

        download(
            url,
            archive,
        )

        extract(archive)

        archive.unlink()

        print(
            f"Finished: {expected_dir}"
        )


if __name__ == "__main__":
    main()
