from __future__ import annotations

import logging
import re
import time
from contextlib import asynccontextmanager
from uuid import uuid4

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api import health
from app.api.v1.router import api_router
from app.core.config import get_settings
from app.core.errors import register_error_handlers
from app.core.logging import configure_logging
from app.core.rate_limit import enforce_rate_limit
from app.core.request_context import (
    clear_request_context,
    set_request_id,
)

settings = get_settings()

configure_logging(
    settings.log_level,
    settings.environment,
)

logger = logging.getLogger(__name__)

REQUEST_ID_PATTERN = re.compile(
    r"^[A-Za-z0-9._:-]{1,128}$"
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(
        "Starting application",
        extra={
            "environment": settings.environment,
        },
    )

    yield

    logger.info(
        "Shutting down application",
        extra={
            "environment": settings.environment,
        },
    )


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    summary="AI-powered image enhancement API",
    description=(
        "Production-oriented REST API for authenticated image uploads, "
        "super-resolution enhancement jobs, image history, job history, "
        "and asynchronous processing."
    ),
    openapi_tags=[
        {
            "name": "health",
            "description": (
                "Liveness and API health endpoints."
            ),
        },
        {
            "name": "authentication",
            "description": (
                "Registration, login, token refresh, logout, "
                "profile, and session management."
            ),
        },
        {
            "name": "images",
            "description": (
                "Authenticated image upload and image history."
            ),
        },
        {
            "name": "enhancements",
            "description": (
                "Asynchronous image enhancement jobs, history, "
                "status, and cancellation."
            ),
        },
    ],
    lifespan=lifespan,
)


@app.middleware("http")
async def request_logging_middleware(
    request: Request,
    call_next,
) -> Response:
    incoming_request_id = request.headers.get("X-Request-ID")

    if (
        incoming_request_id
        and REQUEST_ID_PATTERN.fullmatch(incoming_request_id.strip())
    ):
        request_id = incoming_request_id.strip()
    else:
        request_id = str(uuid4())

    set_request_id(request_id)

    start_time = time.perf_counter()

    try:
        if request.url.path not in {
            "/health",
            "/ready",
            "/api/v1/health",
        }:
            allowed, retry_after = enforce_rate_limit(request)

            if not allowed:
                response = JSONResponse(
                    status_code=429,
                    content={
                        "success": False,
                        "error": {
                            "code": "rate_limit_exceeded",
                            "message": (
                                "Rate limit exceeded. "
                                "Please try again later."
                            ),
                            "details": {},
                        },
                        "request_id": request_id,
                    },
                    headers={
                        "Retry-After": str(retry_after),
                        "X-Request-ID": request_id,
                    },
                )

                logger.warning(
                    "Rate limit exceeded",
                    extra={
                        "environment": settings.environment,
                        "endpoint": request.url.path,
                        "method": request.method,
                        "status": 429,
                    },
                )

                return response

        response = await call_next(request)

        duration_ms = round(
            (time.perf_counter() - start_time) * 1000,
            2,
        )

        logger.info(
            "HTTP request completed",
            extra={
                "environment": settings.environment,
                "endpoint": request.url.path,
                "method": request.method,
                "status": response.status_code,
                "duration_ms": duration_ms,
            },
        )

        response.headers["X-Request-ID"] = request_id

        return response

    except Exception:
        duration_ms = round(
            (time.perf_counter() - start_time) * 1000,
            2,
        )

        logger.exception(
            "HTTP request failed",
            extra={
                "environment": settings.environment,
                "endpoint": request.url.path,
                "method": request.method,
                "duration_ms": duration_ms,
            },
        )

        raise

    finally:
        clear_request_context()


app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

register_error_handlers(app)

app.include_router(health.router)

app.include_router(
    api_router,
    prefix=settings.api_v1_prefix,
)
