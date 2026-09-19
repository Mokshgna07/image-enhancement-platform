from __future__ import annotations

import io
from uuid import uuid4

from fastapi.testclient import TestClient
from PIL import Image as PILImage

from app.api.deps import get_database
from app.api.v1.images import get_storage
from app.db.base import Base
from app.db.models import User
from app.db.models.image import Image
from app.db.session import SessionLocal
from app.main import app
from app.services.storage import FileStorage

from tests.test_auth import TestingSessionLocal, engine


client = TestClient(app)


def override_get_database():
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


app.dependency_overrides[get_database] = override_get_database


class LocalTestStorage(FileStorage):
    def __init__(self):
        super().__init__(root="/tmp/image-enhancement-test-storage")


def override_get_storage():
    return LocalTestStorage()


app.dependency_overrides[get_storage] = override_get_storage


def setup_function():
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)


def teardown_function():
    Base.metadata.drop_all(bind=engine)


def register_and_login(
    email: str = "image-test@example.com",
    password: str = "TestPassword123!",
) -> str:
    register_response = client.post(
        "/api/v1/auth/register",
        json={
            "email": email,
            "password": password,
        },
    )

    assert register_response.status_code == 201

    login_response = client.post(
        "/api/v1/auth/login",
        json={
            "email": email,
            "password": password,
        },
    )

    assert login_response.status_code == 200

    return login_response.json()["access_token"]


def create_test_jpeg() -> bytes:
    image = PILImage.new("RGB", (64, 64), color=(128, 128, 128))

    buffer = io.BytesIO()

    image.save(
        buffer,
        format="JPEG",
        quality=90,
    )

    return buffer.getvalue()


def test_upload_image_requires_authentication():
    response = client.post(
        "/api/v1/images/upload",
        files={
            "file": (
                "test.jpg",
                create_test_jpeg(),
                "image/jpeg",
            )
        },
    )

    assert response.status_code == 401


def test_upload_image_success():
    access_token = register_and_login()

    response = client.post(
        "/api/v1/images/upload",
        headers={
            "Authorization": f"Bearer {access_token}",
        },
        files={
            "file": (
                "test.jpg",
                create_test_jpeg(),
                "image/jpeg",
            )
        },
    )

    assert response.status_code == 201

    data = response.json()

    assert data["original_filename"] == "test.jpg"
    assert data["file_type"] == "image/jpeg"
    assert data["width"] == 64
    assert data["height"] == 64
    assert data["file_size"] > 0
    assert data["id"]


def test_upload_invalid_file_rejected():
    access_token = register_and_login()

    response = client.post(
        "/api/v1/images/upload",
        headers={
            "Authorization": f"Bearer {access_token}",
        },
        files={
            "file": (
                "not-an-image.jpg",
                b"this is not an image",
                "image/jpeg",
            )
        },
    )

    assert response.status_code == 400


def test_get_image_success():
    access_token = register_and_login()

    upload_response = client.post(
        "/api/v1/images/upload",
        headers={
            "Authorization": f"Bearer {access_token}",
        },
        files={
            "file": (
                "test.jpg",
                create_test_jpeg(),
                "image/jpeg",
            )
        },
    )

    assert upload_response.status_code == 201

    image_id = upload_response.json()["id"]

    response = client.get(
        f"/api/v1/images/{image_id}",
        headers={
            "Authorization": f"Bearer {access_token}",
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["id"] == image_id
    assert data["original_filename"] == "test.jpg"
    assert data["width"] == 64
    assert data["height"] == 64


def test_get_image_requires_ownership():
    owner_token = register_and_login(
        email="owner@example.com",
    )

    other_token = register_and_login(
        email="other@example.com",
    )

    upload_response = client.post(
        "/api/v1/images/upload",
        headers={
            "Authorization": f"Bearer {owner_token}",
        },
        files={
            "file": (
                "private.jpg",
                create_test_jpeg(),
                "image/jpeg",
            )
        },
    )

    assert upload_response.status_code == 201

    image_id = upload_response.json()["id"]

    response = client.get(
        f"/api/v1/images/{image_id}",
        headers={
            "Authorization": f"Bearer {other_token}",
        },
    )

    assert response.status_code == 404
