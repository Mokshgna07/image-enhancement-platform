from __future__ import annotations

from uuid import UUID

from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.api.deps import get_database
from app.db.base import Base
from app.db.models import Session as SessionModel
from app.db.models import User
from app.main import app


TEST_DATABASE_URL = "sqlite://"

engine = create_engine(
    TEST_DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)

TestingSessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False,
)


def override_get_database():
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


app.dependency_overrides[get_database] = override_get_database

client = TestClient(app)


def setup_function():
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)


def teardown_function():
    Base.metadata.drop_all(bind=engine)


def register_user(
    email: str = "test@example.com",
    password: str = "TestPassword123!",
):
    return client.post(
        "/api/v1/auth/register",
        json={
            "email": email,
            "password": password,
        },
    )


def login_user(
    email: str = "test@example.com",
    password: str = "TestPassword123!",
):
    return client.post(
        "/api/v1/auth/login",
        json={
            "email": email,
            "password": password,
        },
    )


def test_register_user():
    response = register_user()

    assert response.status_code == 201

    data = response.json()

    assert data["email"] == "test@example.com"
    assert data["email_verified"] is False
    assert data["is_active"] is True
    assert "password" not in data
    assert "password_hash" not in data


def test_duplicate_registration():
    first_response = register_user()
    second_response = register_user()

    assert first_response.status_code == 201
    assert second_response.status_code == 409

    assert second_response.json() == {
        "detail": "An account with this email already exists."
    }


def test_invalid_email_rejected():
    response = register_user(email="not-an-email")

    assert response.status_code == 422


def test_short_password_rejected():
    response = register_user(password="short")

    assert response.status_code == 422


def test_login_success():
    register_user()

    response = login_user()

    assert response.status_code == 200

    data = response.json()

    assert data["token_type"] == "bearer"
    assert data["expires_in"] == 900
    assert data["access_token"]
    assert data["refresh_token"]


def test_wrong_password_rejected():
    register_user()

    response = login_user(password="WrongPassword123!")

    assert response.status_code == 401
    assert response.json() == {
        "detail": "Invalid email or password."
    }


def test_me_requires_authentication():
    response = client.get("/api/v1/auth/me")

    assert response.status_code == 401


def test_me_with_valid_access_token():
    register_user()

    login_response = login_user()
    access_token = login_response.json()["access_token"]

    response = client.get(
        "/api/v1/auth/me",
        headers={
            "Authorization": f"Bearer {access_token}",
        },
    )

    assert response.status_code == 200
    assert response.json()["email"] == "test@example.com"


def test_invalid_access_token_rejected():
    response = client.get(
        "/api/v1/auth/me",
        headers={
            "Authorization": "Bearer definitely-invalid",
        },
    )

    assert response.status_code == 401
    assert response.json() == {
        "detail": "Invalid or expired access token."
    }


def test_refresh_token_rotation():
    register_user()

    login_response = login_user()
    refresh_token = login_response.json()["refresh_token"]

    refresh_response = client.post(
        "/api/v1/auth/refresh",
        json={
            "refresh_token": refresh_token,
        },
    )

    assert refresh_response.status_code == 200

    data = refresh_response.json()

    assert data["access_token"]
    assert data["refresh_token"]
    assert data["refresh_token"] != refresh_token

    old_token_response = client.post(
        "/api/v1/auth/refresh",
        json={
            "refresh_token": refresh_token,
        },
    )

    assert old_token_response.status_code == 401


def test_session_listing():
    register_user()

    login_response = login_user()
    access_token = login_response.json()["access_token"]

    response = client.get(
        "/api/v1/auth/sessions",
        headers={
            "Authorization": f"Bearer {access_token}",
        },
    )

    assert response.status_code == 200

    sessions = response.json()

    assert len(sessions) == 1
    assert sessions[0]["revoked_at"] is None
    assert sessions[0]["user_agent"] is not None


def test_logout_revokes_session():
    register_user()

    login_response = login_user()
    data = login_response.json()

    access_token = data["access_token"]
    refresh_token = data["refresh_token"]

    logout_response = client.post(
        "/api/v1/auth/logout",
        headers={
            "Authorization": f"Bearer {access_token}",
        },
    )

    assert logout_response.status_code == 204

    me_response = client.get(
        "/api/v1/auth/me",
        headers={
            "Authorization": f"Bearer {access_token}",
        },
    )

    assert me_response.status_code == 401

    refresh_response = client.post(
        "/api/v1/auth/refresh",
        json={
            "refresh_token": refresh_token,
        },
    )

    assert refresh_response.status_code == 401


def test_session_revocation():
    register_user()

    login_response = login_user()
    access_token = login_response.json()["access_token"]

    sessions_response = client.get(
        "/api/v1/auth/sessions",
        headers={
            "Authorization": f"Bearer {access_token}",
        },
    )

    session_id = sessions_response.json()[0]["id"]

    assert UUID(session_id)

    revoke_response = client.delete(
        f"/api/v1/auth/sessions/{session_id}",
        headers={
            "Authorization": f"Bearer {access_token}",
        },
    )

    assert revoke_response.status_code == 204

    me_response = client.get(
        "/api/v1/auth/me",
        headers={
            "Authorization": f"Bearer {access_token}",
        },
    )

    assert me_response.status_code == 401

    with TestingSessionLocal() as db:
        session = db.scalar(
            select(SessionModel).where(
                SessionModel.id == UUID(session_id)
            )
        )

        assert session is not None
        assert session.revoked_at is not None


def test_password_hash_is_stored_not_plaintext():
    register_user()

    with TestingSessionLocal() as db:
        user = db.scalar(
            select(User).where(
                User.email == "test@example.com"
            )
        )

        assert user is not None
        assert user.password_hash != "TestPassword123!"
        assert user.password_hash.startswith("$argon2")
