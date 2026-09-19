from __future__ import annotations

import hashlib
import secrets
from datetime import UTC, datetime, timedelta
from uuid import UUID

import jwt
from pwdlib import PasswordHash

from app.core.config import get_settings


password_hash = PasswordHash.recommended()


class SecurityError(Exception):
    """Base exception for authentication security failures."""


class InvalidTokenError(SecurityError):
    """Raised when a JWT is invalid or expired."""


def hash_password(password: str) -> str:
    """Hash a password using the recommended pwdlib algorithm."""
    return password_hash.hash(password)


def verify_password(password: str, hashed_password: str) -> bool:
    """Verify a plaintext password against its stored hash."""
    return password_hash.verify(password, hashed_password)


def create_access_token(user_id: UUID, session_id: UUID) -> str:
    """Create a short-lived JWT access token."""
    settings = get_settings()

    now = datetime.now(UTC)
    expires_at = now + timedelta(
        minutes=settings.access_token_expire_minutes
    )

    payload = {
        "sub": str(user_id),
        "sid": str(session_id),
        "type": "access",
        "iat": now,
        "exp": expires_at,
    }

    return jwt.encode(
        payload,
        settings.jwt_secret_key,
        algorithm=settings.jwt_algorithm,
    )


def create_refresh_token() -> str:
    """Create a cryptographically secure opaque refresh token."""
    return secrets.token_urlsafe(64)


def hash_refresh_token(refresh_token: str) -> str:
    """Hash a refresh token before storing it in the database."""
    return hashlib.sha256(
        refresh_token.encode("utf-8")
    ).hexdigest()


def decode_access_token(token: str) -> dict:
    """Decode and validate an access JWT."""
    settings = get_settings()

    try:
        payload = jwt.decode(
            token,
            settings.jwt_secret_key,
            algorithms=[settings.jwt_algorithm],
        )
    except jwt.PyJWTError as exc:
        raise InvalidTokenError("Invalid or expired access token.") from exc

    if payload.get("type") != "access":
        raise InvalidTokenError("Invalid access token.")

    if not payload.get("sub") or not payload.get("sid"):
        raise InvalidTokenError("Invalid access token.")

    return payload
