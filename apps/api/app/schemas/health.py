from __future__ import annotations

from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str


class ReadinessDependencies(BaseModel):
    database: str
    redis: str
    model: str


class ReadinessResponse(BaseModel):
    status: str
    service: str
    version: str
    dependencies: ReadinessDependencies
