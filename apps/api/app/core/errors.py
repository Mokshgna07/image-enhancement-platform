from __future__ import annotations

import logging
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.core.request_context import get_request_id


logger = logging.getLogger(__name__)


class AppError(Exception):
    def __init__(
        self,
        message: str,
        *,
        status_code: int = 400,
        code: str = "application_error",
        details: dict[str, Any] | None = None,
    ) -> None:
        self.message = message
        self.status_code = status_code
        self.code = code
        self.details = details or {}

        super().__init__(message)


def _error_response(
    *,
    status_code: int,
    code: str,
    message: str,
    details: dict[str, Any] | None = None,
) -> JSONResponse:
    content: dict[str, Any] = {
        "success": False,
        "error": {
            "code": code,
            "message": message,
        },
        "request_id": get_request_id(),
    }

    if details:
        content["error"]["details"] = details

    return JSONResponse(
        status_code=status_code,
        content=content,
    )


async def app_error_handler(
    request: Request,
    exc: AppError,
) -> JSONResponse:
    return _error_response(
        status_code=exc.status_code,
        code=exc.code,
        message=exc.message,
        details=exc.details,
    )


async def http_exception_handler(
    request: Request,
    exc: HTTPException,
) -> JSONResponse:
    detail = exc.detail

    if isinstance(detail, str):
        message = detail
        details = None
    else:
        message = "Request failed."
        details = {"detail": detail}

    code_by_status = {
        400: "bad_request",
        401: "authentication_required",
        403: "forbidden",
        404: "resource_not_found",
        409: "conflict",
        413: "payload_too_large",
        422: "validation_error",
        429: "rate_limit_exceeded",
        500: "internal_server_error",
        503: "service_unavailable",
    }

    return _error_response(
        status_code=exc.status_code,
        code=code_by_status.get(
            exc.status_code,
            "http_error",
        ),
        message=message,
        details=details,
    )


async def validation_exception_handler(
    request: Request,
    exc: RequestValidationError,
) -> JSONResponse:
    return _error_response(
        status_code=422,
        code="validation_error",
        message="Request validation failed.",
        details={
            "fields": exc.errors(),
        },
    )


async def unexpected_error_handler(
    request: Request,
    exc: Exception,
) -> JSONResponse:
    logger.exception(
        "Unhandled exception",
        extra={
            "method": request.method,
            "path": request.url.path,
            "request_id": get_request_id(),
        },
    )

    return _error_response(
        status_code=500,
        code="internal_server_error",
        message="An unexpected error occurred.",
    )


def register_error_handlers(app: FastAPI) -> None:
    app.add_exception_handler(
        AppError,
        app_error_handler,
    )
    app.add_exception_handler(
        HTTPException,
        http_exception_handler,
    )
    app.add_exception_handler(
        RequestValidationError,
        validation_exception_handler,
    )
    app.add_exception_handler(
        Exception,
        unexpected_error_handler,
    )
