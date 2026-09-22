from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import func, select
from sqlalchemy.orm import Session, selectinload

from app.api.deps import get_database
from app.core.request_context import set_image_id, set_job_id, set_user_id
from app.db.models.enhancement_job import EnhancementJob, EnhancementJobStatus
from app.db.models.image import Image
from app.db.models.user import User
from app.dependencies.auth import get_current_user
from app.schemas.enhancements import EnhancementCreate, EnhancementResponse
from app.schemas.pagination import PaginatedResponse
from app.workers.tasks import process_enhancement


router = APIRouter(
    prefix="/enhancements",
    tags=["enhancements"],
)


def _public_job_error(job: EnhancementJob) -> str | None:
    if job.status == EnhancementJobStatus.FAILED:
        return "Enhancement processing failed."

    return None


def _serialize_job(job: EnhancementJob) -> dict:
    return {
        "id": job.id,
        "input_image_id": job.input_image_id,
        "result_image_id": job.result_image_id,
        "scale_factor": job.scale_factor,
        "status": job.status,
        "queued_at": job.queued_at,
        "started_at": job.started_at,
        "completed_at": job.completed_at,
        "failed_at": job.failed_at,
        "error": _public_job_error(job),
        "created_at": job.created_at,
        "updated_at": job.updated_at,
        "processing_metadata": job.processing_metadata,
    }


@router.post(
    "",
    response_model=EnhancementResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
def create_enhancement(
    payload: EnhancementCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_database),
) -> dict:
    set_user_id(str(current_user.id))
    set_image_id(str(payload.image_id))


    if payload.scale_factor not in {2, 4}:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Scale factor must be 2 or 4.",
        )

    image = db.scalar(
        select(Image).where(
            Image.id == payload.image_id,
            Image.user_id == current_user.id,
        )
    )

    if image is None:
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
    set_job_id(str(job.id))

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

    return _serialize_job(job)


@router.get(
    "",
    response_model=PaginatedResponse[EnhancementResponse],
    summary="List enhancement jobs",
    description="Return the authenticated user's enhancement history with pagination.",
)
def list_enhancements(
    page: int = Query(
        default=1,
        ge=1,
        description="Page number, starting at 1.",
    ),
    page_size: int = Query(
        default=20,
        ge=1,
        le=100,
        description="Number of jobs per page. Maximum 100.",
    ),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_database),
) -> dict:
    set_user_id(str(current_user.id))
    user_filter = EnhancementJob.user_id == current_user.id

    total = db.scalar(
        select(func.count())
        .select_from(EnhancementJob)
        .where(user_filter)
    ) or 0

    offset = (page - 1) * page_size

    jobs = db.scalars(
        select(EnhancementJob)
        .where(user_filter)
        .options(
            selectinload(EnhancementJob.processing_metadata),
        )
        .order_by(
            EnhancementJob.created_at.desc(),
            EnhancementJob.id.desc(),
        )
        .offset(offset)
        .limit(page_size)
    ).all()

    return {
        "items": [_serialize_job(job) for job in jobs],
        "page": page,
        "page_size": page_size,
        "total": total,
    }


@router.get(
    "/{job_id}",
    response_model=EnhancementResponse,
    summary="Get enhancement job",
    description="Return one enhancement job belonging to the authenticated user.",
)
def get_enhancement(
    job_id: UUID,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_database),
) -> dict:
    set_user_id(str(current_user.id))
    set_job_id(str(job_id))

    job = db.scalar(
        select(EnhancementJob)
        .where(
            EnhancementJob.id == job_id,
            EnhancementJob.user_id == current_user.id,
        )
        .options(
            selectinload(EnhancementJob.processing_metadata),
        )
    )

    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Enhancement job not found.",
        )

    return _serialize_job(job)


@router.post(
    "/{job_id}/cancel",
    response_model=EnhancementResponse,
    status_code=status.HTTP_200_OK,
    summary="Cancel enhancement job",
    description="Cancel a queued enhancement job belonging to the authenticated user.",
)
def cancel_enhancement(
    job_id: UUID,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_database),
) -> dict:
    set_user_id(str(current_user.id))
    set_job_id(str(job_id))


    job = db.scalar(
        select(EnhancementJob)
        .where(
            EnhancementJob.id == job_id,
            EnhancementJob.user_id == current_user.id,
        )
        .options(
            selectinload(EnhancementJob.processing_metadata),
        )
    )

    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Enhancement job not found.",
        )

    if job.status != EnhancementJobStatus.QUEUED:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                "Only queued enhancement jobs can be canceled. "
                f"Current status: {job.status}."
            ),
        )

    job.status = EnhancementJobStatus.CANCELED
    db.commit()
    db.refresh(job)

    return _serialize_job(job)
