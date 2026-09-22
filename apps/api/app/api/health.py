from __future__ import annotations

from pathlib import Path

import redis
from fastapi import APIRouter, Response, status
from sqlalchemy import text

from app.core.config import get_settings
from app.db.session import engine
from app.schemas.health import (
    HealthResponse,
    ReadinessDependencies,
    ReadinessResponse,
)


router = APIRouter(tags=["health"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Liveness check",
    description="Checks whether the API process is alive.",
)
def health_check() -> HealthResponse:
    settings = get_settings()

    return HealthResponse(
        status="ok",
        service=settings.app_name,
        version=settings.app_version,
    )


@router.get(
    "/ready",
    response_model=ReadinessResponse,
    summary="Readiness check",
    description=(
        "Checks whether the API can serve requests by verifying "
        "PostgreSQL, Redis, and the configured ML checkpoint."
    ),
)
def readiness_check(response: Response) -> ReadinessResponse:
    settings = get_settings()

    database_status = "healthy"
    redis_status = "healthy"
    model_status = "healthy"

    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
    except Exception:
        database_status = "unhealthy"

    try:
        redis_client = redis.Redis.from_url(
            settings.redis_broker_url,
            socket_connect_timeout=1,
            socket_timeout=1,
        )
        redis_client.ping()
        redis_client.close()
    except Exception:
        redis_status = "unhealthy"

    model_path = Path(settings.sr_checkpoint_4x)

    if not model_path.is_file():
        model_status = "unhealthy"

    dependencies = ReadinessDependencies(
        database=database_status,
        redis=redis_status,
        model=model_status,
    )

    all_healthy = all(
        value == "healthy"
        for value in (
            database_status,
            redis_status,
            model_status,
        )
    )

    if not all_healthy:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

        return ReadinessResponse(
            status="not_ready",
            service=settings.app_name,
            version=settings.app_version,
            dependencies=dependencies,
        )

    return ReadinessResponse(
        status="ready",
        service=settings.app_name,
        version=settings.app_version,
        dependencies=dependencies,
    )
