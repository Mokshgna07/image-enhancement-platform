from __future__ import annotations

import os
import tempfile
from pathlib import Path
from uuid import UUID, uuid4

from app.core.config import get_settings


class StorageError(RuntimeError):
    """Base exception for storage failures."""


class FileStorage:
    """Local filesystem storage for uploaded and generated images."""

    ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}

    def __init__(self, root: str | Path | None = None) -> None:
        settings = get_settings()

        self.root = Path(
            root if root is not None else settings.storage_root
        ).expanduser()

        self.image_root = self.root / "images"
        self.image_root.mkdir(
            parents=True,
            exist_ok=True,
        )

    def build_image_path(
        self,
        image_id: UUID,
        extension: str,
    ) -> Path:
        extension = extension.lower().lstrip(".")

        if extension not in self.ALLOWED_EXTENSIONS:
            raise StorageError(
                f"Unsupported storage extension: {extension}"
            )

        return self.image_root / f"{image_id}.{extension}"

    def build_output_path(
        self,
        image_id: UUID,
    ) -> Path:
        return self.image_root / f"{image_id}.png"

    def create_temp_file(self) -> Path:
        try:
            fd, path = tempfile.mkstemp(
                prefix="upload-",
                suffix=".tmp",
                dir=self.image_root,
            )
            os.close(fd)
            return Path(path)

        except OSError as exc:
            raise StorageError(
                "Failed to create temporary upload file."
            ) from exc

    def move_into_place(
        self,
        source: Path,
        destination: Path,
    ) -> None:
        try:
            destination.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            os.replace(
                source,
                destination,
            )

        except OSError as exc:
            raise StorageError(
                f"Failed to store file: {destination}"
            ) from exc

    def delete(
        self,
        path: str | Path,
    ) -> None:
        target = Path(path)

        try:
            target.unlink(missing_ok=True)

        except OSError as exc:
            raise StorageError(
                f"Failed to delete stored file: {target}"
            ) from exc

    def exists(
        self,
        path: str | Path,
    ) -> bool:
        return Path(path).is_file()

    def resolve_path(
        self,
        path: str | Path,
    ) -> Path:
        """Resolve a stored path independently of the process working directory."""

        target = Path(path)

        if target.is_absolute():
            return target

        storage_name = self.root.name

        if target.parts and target.parts[0] == storage_name:
            target = Path(*target.parts[1:])

        return self.root / target

    @staticmethod
    def new_image_id() -> UUID:
        return uuid4()
