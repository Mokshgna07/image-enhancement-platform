from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict

from app.db.models.enhancement_job import EnhancementJobStatus


class EnhancementCreate(BaseModel):
    image_id: UUID
    scale_factor: int = 4


class EnhancementResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    input_image_id: UUID
    result_image_id: UUID | None
    scale_factor: int
    status: EnhancementJobStatus
    queued_at: datetime
    started_at: datetime | None
    completed_at: datetime | None
    failed_at: datetime | None
    error_message: str | None
    created_at: datetime
    updated_at: datetime
