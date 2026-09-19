from __future__ import annotations

from datetime import UTC, datetime, timedelta
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.security import (
    create_access_token,
    create_refresh_token,
    hash_password,
    hash_refresh_token,
    verify_password,
)
from app.db.models.session import Session as SessionModel
from app.db.models.user import User


class AuthenticationError(Exception):
    """Base authentication service error."""


class DuplicateEmailError(AuthenticationError):
    """Raised when an email is already registered."""


class InvalidCredentialsError(AuthenticationError):
    """Raised when login credentials are invalid."""


class InvalidRefreshTokenError(AuthenticationError):
    """Raised when a refresh token cannot be used."""


class AuthenticationService:
    """Business logic for users, sessions, and authentication tokens."""

    def __init__(self, db: Session) -> None:
        self.db = db
        self.settings = get_settings()

    def register(
        self,
        *,
        email: str,
        password: str,
    ) -> User:
        normalized_email = email.strip().lower()

        existing_user = self.db.scalar(
            select(User).where(User.email == normalized_email)
        )

        if existing_user is not None:
            raise DuplicateEmailError(
                "An account with this email already exists."
            )

        user = User(
            email=normalized_email,
            password_hash=hash_password(password),
            email_verified=False,
            is_active=True,
        )

        self.db.add(user)
        self.db.commit()
        self.db.refresh(user)

        return user

    def authenticate(
        self,
        *,
        email: str,
        password: str,
        user_agent: str | None = None,
        ip_address: str | None = None,
    ) -> tuple[User, SessionModel, str, str]:
        normalized_email = email.strip().lower()

        user = self.db.scalar(
            select(User).where(User.email == normalized_email)
        )

        if user is None or not verify_password(
            password,
            user.password_hash,
        ):
            raise InvalidCredentialsError("Invalid email or password.")

        if not user.is_active:
            raise InvalidCredentialsError("Invalid email or password.")

        return self._create_session(
            user=user,
            user_agent=user_agent,
            ip_address=ip_address,
        )

    def _create_session(
        self,
        *,
        user: User,
        user_agent: str | None,
        ip_address: str | None,
    ) -> tuple[User, SessionModel, str, str]:
        refresh_token = create_refresh_token()

        expires_at = datetime.now(UTC) + timedelta(
            days=self.settings.refresh_token_expire_days
        )

        session = SessionModel(
            user_id=user.id,
            refresh_token_hash=hash_refresh_token(refresh_token),
            expires_at=expires_at,
            last_used_at=datetime.now(UTC),
            user_agent=user_agent,
            ip_address=ip_address,
        )

        self.db.add(session)
        self.db.commit()
        self.db.refresh(session)

        access_token = create_access_token(
            user.id,
            session.id,
        )

        return user, session, access_token, refresh_token

    def refresh(
        self,
        *,
        refresh_token: str,
        user_agent: str | None = None,
        ip_address: str | None = None,
    ) -> tuple[User, SessionModel, str, str]:
        token_hash = hash_refresh_token(refresh_token)

        session = self.db.scalar(
            select(SessionModel).where(
                SessionModel.refresh_token_hash == token_hash
            )
        )

        if session is None:
            raise InvalidRefreshTokenError("Invalid refresh token.")

        now = datetime.now(UTC)

        expires_at = session.expires_at

        if expires_at.tzinfo is None:
            expires_at = expires_at.replace(tzinfo=UTC)

        if session.revoked_at is not None:
            raise InvalidRefreshTokenError("Invalid refresh token.")

        if expires_at <= now:
            raise InvalidRefreshTokenError("Refresh token has expired.")

        user = self.db.get(User, session.user_id)

        if user is None or not user.is_active:
            raise InvalidRefreshTokenError("Invalid refresh token.")

        # Rotate the refresh token.
        new_refresh_token = create_refresh_token()
        session.refresh_token_hash = hash_refresh_token(new_refresh_token)
        session.last_used_at = now

        if user_agent is not None:
            session.user_agent = user_agent

        if ip_address is not None:
            session.ip_address = ip_address

        self.db.commit()
        self.db.refresh(session)

        access_token = create_access_token(
            user.id,
            session.id,
        )

        return user, session, access_token, new_refresh_token

    def revoke_session(
        self,
        *,
        user_id: UUID,
        session_id: UUID,
    ) -> None:
        session = self.db.scalar(
            select(SessionModel).where(
                SessionModel.id == session_id,
                SessionModel.user_id == user_id,
            )
        )

        if session is None:
            raise InvalidRefreshTokenError("Session not found.")

        if session.revoked_at is None:
            session.revoked_at = datetime.now(UTC)
            self.db.commit()

    def revoke_all_sessions(
        self,
        *,
        user_id: UUID,
    ) -> None:
        sessions = self.db.scalars(
            select(SessionModel).where(
                SessionModel.user_id == user_id,
                SessionModel.revoked_at.is_(None),
            )
        ).all()

        now = datetime.now(UTC)

        for session in sessions:
            session.revoked_at = now

        self.db.commit()
