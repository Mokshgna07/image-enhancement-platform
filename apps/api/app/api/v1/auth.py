from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.deps import get_database
from app.core.config import get_settings
from app.core.security import InvalidTokenError, decode_access_token
from app.db.models.session import Session as SessionModel
from app.db.models.user import User
from app.dependencies.auth import bearer_scheme, get_current_user
from app.schemas.auth import (
    LoginRequest,
    RefreshTokenRequest,
    RegisterRequest,
    SessionResponse,
    TokenResponse,
    UserResponse,
)
from app.services.authentication import (
    AuthenticationService,
    DuplicateEmailError,
    InvalidCredentialsError,
    InvalidRefreshTokenError,
)


router = APIRouter(prefix="/auth", tags=["authentication"])


def get_auth_service(
    db: Session = Depends(get_database),
) -> AuthenticationService:
    return AuthenticationService(db)


def _client_ip(request: Request) -> str | None:
    if request.client is None:
        return None
    return request.client.host


def _user_agent(request: Request) -> str | None:
    return request.headers.get("user-agent")


def _token_response(
    access_token: str,
    refresh_token: str,
) -> TokenResponse:
    settings = get_settings()

    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        token_type="bearer",
        expires_in=settings.access_token_expire_minutes * 60,
    )


@router.post(
    "/register",
    response_model=UserResponse,
    status_code=status.HTTP_201_CREATED,
)
def register(
    payload: RegisterRequest,
    service: AuthenticationService = Depends(get_auth_service),
) -> User:
    try:
        return service.register(
            email=str(payload.email),
            password=payload.password,
        )
    except DuplicateEmailError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc


@router.post(
    "/login",
    response_model=TokenResponse,
)
def login(
    payload: LoginRequest,
    request: Request,
    service: AuthenticationService = Depends(get_auth_service),
) -> TokenResponse:
    try:
        _, _, access_token, refresh_token = service.authenticate(
            email=str(payload.email),
            password=payload.password,
            user_agent=_user_agent(request),
            ip_address=_client_ip(request),
        )
    except InvalidCredentialsError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(exc),
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc

    return _token_response(access_token, refresh_token)


@router.post(
    "/refresh",
    response_model=TokenResponse,
)
def refresh(
    payload: RefreshTokenRequest,
    request: Request,
    service: AuthenticationService = Depends(get_auth_service),
) -> TokenResponse:
    try:
        _, _, access_token, new_refresh_token = service.refresh(
            refresh_token=payload.refresh_token,
            user_agent=_user_agent(request),
            ip_address=_client_ip(request),
        )
    except InvalidRefreshTokenError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(exc),
        ) from exc

    return _token_response(access_token, new_refresh_token)


@router.post(
    "/logout",
    status_code=status.HTTP_204_NO_CONTENT,
)
def logout(
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_database),
) -> None:
    try:
        payload = decode_access_token(credentials.credentials)
        session_id = UUID(payload["sid"])
    except (InvalidTokenError, ValueError, KeyError):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired access token.",
            headers={"WWW-Authenticate": "Bearer"},
        ) from None

    service = AuthenticationService(db)
    service.revoke_session(
        user_id=current_user.id,
        session_id=session_id,
    )


@router.get(
    "/me",
    response_model=UserResponse,
)
def me(
    current_user: User = Depends(get_current_user),
) -> User:
    return current_user


@router.get(
    "/sessions",
    response_model=list[SessionResponse],
)
def sessions(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_database),
) -> list[SessionModel]:
    return list(
        db.scalars(
            select(SessionModel)
            .where(SessionModel.user_id == current_user.id)
            .order_by(SessionModel.created_at.desc())
        ).all()
    )


@router.delete(
    "/sessions/{session_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
def revoke_session(
    session_id: UUID,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_database),
) -> None:
    service = AuthenticationService(db)

    try:
        service.revoke_session(
            user_id=current_user.id,
            session_id=session_id,
        )
    except InvalidRefreshTokenError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
