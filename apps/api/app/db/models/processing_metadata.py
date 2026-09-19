from __future__ import annotations

from datetime import datetime
from uuid import UUID, uuid4

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base


class ProcessingMetadata(Base):
    __tablename__ = "processing_metadata"

    id: Mapped[UUID] = mapped_column(
        primary_key=True,
        default=uuid4,
    )

    job_id: Mapped[UUID] = mapped_column(
        ForeignKey("enhancement_jobs.id", ondelete="CASCADE"),
        unique=True,
        nullable=False,
        index=True,
    )

    model_version: Mapped[str] = mapped_column(
        String(128),
        nullable=False,
    )

    scale_factor: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    input_width: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    input_height: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    output_width: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )

    output_height: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )

    processing_time_ms: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )

    device: Mapped[str | None] = mapped_column(
        String(32),
        nullable=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    job: Mapped["EnhancementJob"] = relationship(
        back_populates="processing_metadata",
    )
