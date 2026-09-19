from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.api.deps import get_database
from app.db.models.enhancement_job import EnhancementJob, EnhancementJobStatus
from app.db.models.image import Image
from app.db.models.user import User
from app.dependencies.auth import get_current_user
from app.schemas.enhancements import EnhancementCreate, EnhancementResponse
from app.workers.tasks import process_enhancement


router = APIRouter(
    prefix="/enhancements",
    tags=["enhancements"],
)


@router.post(
    "",
    response_model=EnhancementResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
def create_enhancement(
    payload: EnhancementCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_database),
) -> EnhancementJob:
    if payload.scale_factor not in {2, 4}:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Scale factor must be 2 or 4.",
        )

    image = db.get(Image, payload.image_id)

    if image is None or image.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Image not found.",
        )

    job = EnhancementJob(
        user_id=current_user.id,
        input_image_id=image.id,
        scale_factor=payload.scale_factor,
    )

    db.add(job)
    db.commit()
    db.refresh(job)

    try:
        process_enhancement.delay(str(job.id))
    except Exception as exc:
        job.status = EnhancementJobStatus.FAILED
        job.error_message = f"Failed to queue enhancement job: {exc}"[:4000]
        db.commit()

        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Enhancement queue is unavailable.",
        ) from exc

    return job


@router.get(
    "/{job_id}",
    response_model=EnhancementResponse,
)
def get_enhancement(
    job_id: UUID,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_database),
) -> EnhancementJob:
    job = db.get(EnhancementJob, job_id)

    if job is None or job.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Enhancement job not found.",
        )

    return job
