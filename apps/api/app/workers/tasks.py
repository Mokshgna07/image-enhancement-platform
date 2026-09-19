from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID

from celery import Task
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.db.models.enhancement_job import EnhancementJob, EnhancementJobStatus
from app.db.models.image import Image
from app.db.models.processing_metadata import ProcessingMetadata
from app.db.session import SessionLocal
from app.services.storage import FileStorage
from app.workers.celery_app import celery_app
from ml.src.super_resolution.inference import build_service_from_environment


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class EnhancementTask(Task):
    _inference_service = None

    @property
    def inference_service(self):
        if self._inference_service is None:
            settings = get_settings()

            os.environ["SR_CONFIG_PATH"] = settings.sr_config_path
            os.environ["SR_CHECKPOINT_4X"] = settings.sr_checkpoint_4x

            if settings.sr_checkpoint_2x:
                os.environ["SR_CHECKPOINT_2X"] = settings.sr_checkpoint_2x
            else:
                os.environ.pop("SR_CHECKPOINT_2X", None)

            os.environ["SR_DEVICE"] = settings.model_device
            os.environ["SR_MODEL_VERSION"] = settings.model_version

            self._inference_service = build_service_from_environment()

        return self._inference_service


@celery_app.task(
    bind=True,
    base=EnhancementTask,
    name="enhancement.process_image",
)
def process_enhancement(
    self: EnhancementTask,
    job_id: str,
) -> None:
    db: Session = SessionLocal()
    storage = FileStorage()
    job: EnhancementJob | None = None
    result_path: Path | None = None

    try:
        job = db.get(EnhancementJob, UUID(job_id))

        if job is None:
            raise ValueError(f"Enhancement job not found: {job_id}")

        if job.status not in {
            EnhancementJobStatus.QUEUED,
            EnhancementJobStatus.PROCESSING,
        }:
            return

        job.status = EnhancementJobStatus.PROCESSING
        job.started_at = utc_now()
        job.error_message = None
        db.commit()

        input_image = db.get(Image, job.input_image_id)

        if input_image is None:
            raise ValueError(
                f"Input image not found: {job.input_image_id}"
            )

        input_path = storage.resolve_path(input_image.storage_path)

        if not input_path.is_file():
            raise FileNotFoundError(
                f"Input image file not found: {input_path}"
            )

        result_image_id = storage.new_image_id()
        result_path = storage.build_output_path(result_image_id)

        inference_result = self.inference_service.enhance(
            input_path,
            result_path,
            scale=job.scale_factor,
        )

        result_image = Image(
            id=result_image_id,
            user_id=job.user_id,
            source_image_id=input_image.id,
            original_filename=(
                f"{Path(input_image.original_filename).stem}"
                f"_x{job.scale_factor}.png"
            ),
            file_type="image/png",
            file_size=result_path.stat().st_size,
            width=inference_result.output_width,
            height=inference_result.output_height,
            storage_path=str(result_path),
        )

        db.add(result_image)

        metadata = ProcessingMetadata(
            job_id=job.id,
            model_version=inference_result.model_version,
            scale_factor=inference_result.scale,
            input_width=inference_result.input_width,
            input_height=inference_result.input_height,
            output_width=inference_result.output_width,
            output_height=inference_result.output_height,
            processing_time_ms=inference_result.processing_time_ms,
            device=inference_result.device,
        )

        db.add(metadata)

        job.result_image_id = result_image_id
        job.status = EnhancementJobStatus.COMPLETED
        job.completed_at = utc_now()
        job.error_message = None

        db.commit()

    except Exception as exc:
        db.rollback()

        if job is not None:
            job.status = EnhancementJobStatus.FAILED
            job.failed_at = utc_now()
            job.error_message = str(exc)[:4000]
            db.commit()

        if result_path is not None:
            storage.delete(result_path)

        raise

    finally:
        db.close()
