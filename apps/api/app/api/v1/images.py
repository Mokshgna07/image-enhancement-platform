from __future__ import annotations

from pathlib import Path
from uuid import UUID

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from sqlalchemy.orm import Session

from app.api.deps import get_database
from app.db.models.image import Image
from app.db.models.user import User
from app.dependencies.auth import get_current_user
from app.schemas.images import ImageResponse
from app.services.image_processing import (
    ImageProcessingService,
    ImageUploadValidationError,
)
from app.services.storage import FileStorage, StorageError


router = APIRouter(
    prefix="/images",
    tags=["images"],
)


def get_storage() -> FileStorage:
    return FileStorage()


def get_image_processing_service() -> ImageProcessingService:
    return ImageProcessingService()


def _copy_upload_to_temp(
    upload: UploadFile,
    storage: FileStorage,
) -> Path:
    temp_path = storage.create_temp_file()

    try:
        total_size = 0
        chunk_size = 1024 * 1024

        with temp_path.open("wb") as output:
            while True:
                chunk = upload.file.read(chunk_size)

                if not chunk:
                    break

                total_size += len(chunk)

                if total_size > ImageProcessingService().max_upload_size_bytes:
                    raise ImageUploadValidationError(
                        "Uploaded file exceeds the maximum allowed size."
                    )

                output.write(chunk)

        return temp_path

    except Exception:
        storage.delete(temp_path)
        raise


@router.post(
    "/upload",
    response_model=ImageResponse,
    status_code=status.HTTP_201_CREATED,
)
def upload_image(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_database),
    storage: FileStorage = Depends(get_storage),
    processor: ImageProcessingService = Depends(
        get_image_processing_service,
    ),
) -> Image:
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="A filename is required.",
        )

    temp_path: Path | None = None
    final_path: Path | None = None

    try:
        temp_path = storage.create_temp_file()

        total_size = 0
        chunk_size = 1024 * 1024

        with temp_path.open("wb") as output:
            while True:
                chunk = file.file.read(chunk_size)

                if not chunk:
                    break

                total_size += len(chunk)

                if total_size > processor.max_upload_size_bytes:
                    raise HTTPException(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        detail="Uploaded file exceeds the maximum allowed size.",
                    )

                output.write(chunk)

        validated = processor.validate_file(
            temp_path,
            declared_content_type=file.content_type,
            original_filename=file.filename,
        )

        image_id = storage.new_image_id()

        final_path = storage.build_image_path(
            image_id,
            validated.extension,
        )

        storage.move_into_place(
            temp_path,
            final_path,
        )

        temp_path = None

        image = Image(
            id=image_id,
            user_id=current_user.id,
            original_filename=file.filename,
            file_type=validated.mime_type,
            file_size=total_size,
            width=validated.width,
            height=validated.height,
            storage_path=str(final_path),
        )

        db.add(image)
        db.commit()
        db.refresh(image)

        return image

    except HTTPException:
        if temp_path is not None:
            storage.delete(temp_path)

        if final_path is not None:
            storage.delete(final_path)

        raise

    except ImageUploadValidationError as exc:
        if temp_path is not None:
            storage.delete(temp_path)

        if final_path is not None:
            storage.delete(final_path)

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    except StorageError as exc:
        if temp_path is not None:
            storage.delete(temp_path)

        if final_path is not None:
            storage.delete(final_path)

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to store uploaded image.",
        ) from exc

    except Exception:
        if temp_path is not None:
            storage.delete(temp_path)

        if final_path is not None:
            storage.delete(final_path)

        db.rollback()

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process uploaded image.",
        )
@router.get("/{image_id}", response_model=ImageResponse)
def get_image(
    image_id: UUID,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_database),
) -> Image:
    image = db.get(Image, image_id)

    if image is None or image.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Image not found.",
        )

    return image
