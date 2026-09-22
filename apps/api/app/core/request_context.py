from __future__ import annotations

from contextvars import ContextVar


_request_id: ContextVar[str | None] = ContextVar(
    "request_id",
    default=None,
)

_user_id: ContextVar[str | None] = ContextVar(
    "user_id",
    default=None,
)

_job_id: ContextVar[str | None] = ContextVar(
    "job_id",
    default=None,
)

_image_id: ContextVar[str | None] = ContextVar(
    "image_id",
    default=None,
)


def set_request_id(request_id: str) -> None:
    _request_id.set(request_id)


def get_request_id() -> str | None:
    return _request_id.get()


def set_user_id(user_id: str) -> None:
    _user_id.set(user_id)


def get_user_id() -> str | None:
    return _user_id.get()


def set_job_id(job_id: str) -> None:
    _job_id.set(job_id)


def get_job_id() -> str | None:
    return _job_id.get()


def set_image_id(image_id: str) -> None:
    _image_id.set(image_id)


def get_image_id() -> str | None:
    return _image_id.get()


def clear_request_context() -> None:
    _request_id.set(None)
    _user_id.set(None)
    _job_id.set(None)
    _image_id.set(None)
