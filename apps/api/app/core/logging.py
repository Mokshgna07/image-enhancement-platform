from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any

from app.core.request_context import (
    get_image_id,
    get_job_id,
    get_request_id,
    get_user_id,
)


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "service": "image-enhancement-api",
            "logger": record.name,
            "message": record.getMessage(),
        }

        context_fields = {
            "request_id": get_request_id(),
            "user_id": get_user_id(),
            "job_id": get_job_id(),
            "image_id": get_image_id(),
        }

        for field, value in context_fields.items():
            if value is not None:
                payload[field] = value

        for field in (
            "environment",
            "endpoint",
            "method",
            "status",
            "duration_ms",
            "error_code",
        ):
            value = getattr(record, field, None)

            if value is not None:
                payload[field] = value

        if record.exc_info:
            payload["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
            }

        return json.dumps(
            payload,
            ensure_ascii=False,
            default=str,
        )


def configure_logging(
    level: str = "INFO",
    environment: str = "development",
) -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())

    root_logger = logging.getLogger()

    root_logger.handlers.clear()
    root_logger.setLevel(
        getattr(logging, level.upper(), logging.INFO),
    )
    root_logger.addHandler(handler)

    logging.captureWarnings(True)
