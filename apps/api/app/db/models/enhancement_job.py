from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from uuid import UUID, uuid4

from sqlalchemy import DateTime, Enum, ForeignKey, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base


class EnhancementJobStatus(StrEnum):
    QUEUED = "QUEUED"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class EnhancementJob(Base):
    __tablename__ = "enhancement_jobs"

    id: Mapped[UUID] = mapped_column(
        primary_key=True,
        default=uuid4,
    )

    user_id: Mapped[UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    input_image_id: Mapped[UUID] = mapped_column(
        ForeignKey("images.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    result_image_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("images.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )

    scale_factor: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    status: Mapped[EnhancementJobStatus] = mapped_column(
        Enum(
            EnhancementJobStatus,
            name="enhancement_job_status",
            native_enum=True,
        ),
        nullable=False,
        default=EnhancementJobStatus.QUEUED,
        index=True,
    )

    queued_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    started_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    failed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    error_message: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )

    input_image: Mapped["Image"] = relationship(
        foreign_keys=[input_image_id],
    )

    result_image: Mapped["Image | None"] = relationship(
        foreign_keys=[result_image_id],
    )

    processing_metadata: Mapped["ProcessingMetadata | None"] = relationship(
        back_populates="job",
        uselist=False,
        cascade="all, delete-orphan",
    )
