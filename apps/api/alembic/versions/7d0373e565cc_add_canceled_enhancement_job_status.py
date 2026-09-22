"""add canceled enhancement job status

Revision ID: 7d0373e565cc
Revises: cbb17fb314a0
Create Date: 2026-09-20 08:04:20.987064

"""

from typing import Sequence, Union

from alembic import op


revision: str = "7d0373e565cc"
down_revision: Union[str, Sequence[str], None] = "cbb17fb314a0"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute(
        """
        ALTER TYPE enhancement_job_status
        ADD VALUE IF NOT EXISTS 'CANCELED'
        """
    )


def downgrade() -> None:
    # PostgreSQL does not support removing an individual value
    # from an enum type safely with ALTER TYPE.
    pass
