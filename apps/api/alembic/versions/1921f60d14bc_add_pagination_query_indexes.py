"""add pagination query indexes

Revision ID: 1921f60d14bc
Revises: 7d0373e565cc
Create Date: 2026-09-22
"""

from typing import Sequence, Union

from alembic import op


revision: str = "1921f60d14bc"
down_revision: Union[str, Sequence[str], None] = "7d0373e565cc"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_index(
        "ix_images_user_created_id",
        "images",
        ["user_id", "created_at", "id"],
    )

    op.create_index(
        "ix_enhancement_jobs_user_created_id",
        "enhancement_jobs",
        ["user_id", "created_at", "id"],
    )

    op.create_index(
        "ix_sessions_user_created_id",
        "sessions",
        ["user_id", "created_at", "id"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_sessions_user_created_id",
        table_name="sessions",
    )

    op.drop_index(
        "ix_enhancement_jobs_user_created_id",
        table_name="enhancement_jobs",
    )

    op.drop_index(
        "ix_images_user_created_id",
        table_name="images",
    )
