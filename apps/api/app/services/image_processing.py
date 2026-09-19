from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from app.core.config import get_settings


class ImageProcessingError(RuntimeError):
    """Base exception for image processing failures."""


class ImageUploadValidationError(ImageProcessingError):
    """Raised when an uploaded image is invalid."""


@dataclass(frozen=True)
class ValidatedImage:
    format: str
    mime_type: str
    extension: str
    width: int
    height: int


class ImageProcessingService:
    """Validate uploaded images and extract trusted metadata."""

    ALLOWED_TYPES = {
        "JPEG": ("image/jpeg", {"jpg", "jpeg"}),
        "PNG": ("image/png", {"png"}),
        "WEBP": ("image/webp", {"webp"}),
    }

    def __init__(
        self,
        max_upload_size_bytes: int | None = None,
        max_image_pixels: int | None = None,
    ) -> None:
        settings = get_settings()

        self.max_upload_size_bytes = (
            max_upload_size_bytes
            if max_upload_size_bytes is not None
            else settings.max_upload_size_bytes
        )

        self.max_image_pixels = (
            max_image_pixels
            if max_image_pixels is not None
            else settings.max_image_pixels
        )

    def validate_file(
        self,
        path: str | Path,
        *,
        declared_content_type: str | None = None,
        original_filename: str | None = None,
    ) -> ValidatedImage:
        path = Path(path)

        if not path.exists() or not path.is_file():
            raise ImageUploadValidationError(
                "Uploaded file does not exist."
            )

        try:
            file_size = path.stat().st_size

        except OSError as exc:
            raise ImageUploadValidationError(
                "Unable to inspect uploaded file."
            ) from exc

        if file_size <= 0:
            raise ImageUploadValidationError(
                "Uploaded file is empty."
            )

        if file_size > self.max_upload_size_bytes:
            raise ImageUploadValidationError(
                "Uploaded file exceeds the maximum allowed size."
            )

        extension = ""

        if original_filename:
            extension = (
                Path(original_filename)
                .suffix
                .lower()
                .lstrip(".")
            )

        if extension not in {
            "jpg",
            "jpeg",
            "png",
            "webp",
        }:
            raise ImageUploadValidationError(
                "Unsupported image extension."
            )

        if declared_content_type not in {
            "image/jpeg",
            "image/png",
            "image/webp",
        }:
            raise ImageUploadValidationError(
                "Unsupported image MIME type."
            )

        try:
            # First pass checks the image structure.
            with Image.open(path) as image:
                image.verify()

            # Second pass extracts metadata and fully loads the image.
            with Image.open(path) as image:
                image.load()

                image_format = image.format
                width, height = image.size

                if image_format not in self.ALLOWED_TYPES:
                    raise ImageUploadValidationError(
                        "Unsupported image format."
                    )

                expected_mime, extensions = self.ALLOWED_TYPES[
                    image_format
                ]

                if declared_content_type != expected_mime:
                    raise ImageUploadValidationError(
                        "Declared MIME type does not match the image format."
                    )

                if extension not in extensions:
                    raise ImageUploadValidationError(
                        "File extension does not match the image format."
                    )

                if width <= 0 or height <= 0:
                    raise ImageUploadValidationError(
                        "Image dimensions must be positive."
                    )

                if width * height > self.max_image_pixels:
                    raise ImageUploadValidationError(
                        "Image exceeds the maximum supported pixel count."
                    )

                return ValidatedImage(
                    format=image_format,
                    mime_type=expected_mime,
                    extension=extension,
                    width=width,
                    height=height,
                )

        except ImageUploadValidationError:
            raise

        except UnidentifiedImageError as exc:
            raise ImageUploadValidationError(
                "Uploaded file is not a valid image."
            ) from exc

        except Exception as exc:
            raise ImageUploadValidationError(
                "Failed to validate uploaded image."
            ) from exc
