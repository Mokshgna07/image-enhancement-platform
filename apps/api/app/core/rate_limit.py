from __future__ import annotations

import time

import redis
from fastapi import Request

from app.core.config import get_settings


settings = get_settings()


class RateLimiter:
    def __init__(self) -> None:
        self.client = redis.Redis.from_url(
            settings.rate_limit_redis_url,
            decode_responses=True,
        )

    def check(self, key: str) -> tuple[bool, int]:
        if not settings.rate_limit_enabled:
            return True, 0

        now = int(time.time())
        window = settings.rate_limit_window_seconds
        window_start = now - (now % window)

        redis_key = f"rate-limit:{key}:{window_start}"

        try:
            pipe = self.client.pipeline()
            pipe.incr(redis_key)
            pipe.expire(redis_key, window + 1)

            results = pipe.execute()
            request_count = int(results[0])

        except redis.RedisError:
            # Fail open if Redis is temporarily unavailable.
            return True, 0

        if request_count <= settings.rate_limit_requests:
            return True, 0

        retry_after = window - (now % window)

        return False, max(retry_after, 1)

    def close(self) -> None:
        self.client.close()


def get_client_identifier(request: Request) -> str:
    if request.client is None:
        return "unknown"

    return request.client.host


def enforce_rate_limit(request: Request) -> tuple[bool, int]:
    limiter = RateLimiter()

    try:
        identifier = get_client_identifier(request)

        return limiter.check(identifier)

    finally:
        limiter.close()
