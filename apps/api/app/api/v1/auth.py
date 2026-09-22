from __future__ import annotations

from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from fastapi.security import HTTPAuthorizationCredentials
from sqlalchemy import func, select
from app.schemas.pagination import PaginatedResponse
from sqlalchemy.orm import Session

from app.api.deps import get_database
from app.core.config import get_settings
from app.core.request_context import set_user_id
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
    set_user_id(str(current_user.id))
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
    set_user_id(str(current_user.id))
    return current_user


@router.get(
    "/sessions",
    response_model=PaginatedResponse[SessionResponse],
    summary="List active sessions",
    description="Return the authenticated user's sessions with pagination.",
)
def sessions(
    page: int = Query(
        default=1,
        ge=1,
        description="Page number, starting at 1.",
    ),
    page_size: int = Query(
        default=20,
        ge=1,
        le=100,
        description="Number of sessions per page. Maximum 100.",
    ),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_database),
) -> dict:
    set_user_id(str(current_user.id))
    user_filter = SessionModel.user_id == current_user.id

    total = db.scalar(
        select(func.count())
        .select_from(SessionModel)
        .where(user_filter)
    ) or 0

    offset = (page - 1) * page_size

    sessions = db.scalars(
        select(SessionModel)
        .where(user_filter)
        .order_by(
            SessionModel.created_at.desc(),
            SessionModel.id.desc(),
        )
        .offset(offset)
        .limit(page_size)
    ).all()

    return {
        "items": list(sessions),
        "page": page,
        "page_size": page_size,
        "total": total,
    }

@router.delete(
    "/sessions/{session_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
def revoke_session(
    session_id: UUID,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_database),
) -> None:
    set_user_id(str(current_user.id))
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
