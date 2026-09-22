from __future__ import annotations

from typing import Generic, TypeVar

from pydantic import BaseModel, Field


T = TypeVar("T")


class PaginationParams(BaseModel):
    page: int = Field(
        default=1,
        ge=1,
        description="Page number, starting at 1.",
    )
    page_size: int = Field(
        default=20,
        ge=1,
        le=100,
        description="Number of items per page. Maximum 100.",
    )


class PaginatedResponse(BaseModel, Generic[T]):
    items: list[T]
    page: int
    page_size: int
    total: int
