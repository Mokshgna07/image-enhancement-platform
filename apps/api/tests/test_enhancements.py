from __future__ import annotations

import io
from unittest.mock import patch
from uuid import uuid4

from fastapi.testclient import TestClient
from PIL import Image as PILImage

from app.api.deps import get_database
from app.db.base import Base
from app.db.models.enhancement_job import EnhancementJob, EnhancementJobStatus
from app.db.models.image import Image
from app.main import app

from tests.test_auth import TestingSessionLocal, engine


client = TestClient(app)


def override_get_database():
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


app.dependency_overrides[get_database] = override_get_database


def setup_function():
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)


def teardown_function():
    Base.metadata.drop_all(bind=engine)


def register_and_login(
    email: str = "enhancement-test@example.com",
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


def upload_test_image(access_token: str) -> str:
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

    return response.json()["id"]


def test_create_enhancement_requires_authentication():
    response = client.post(
        "/api/v1/enhancements",
        json={
            "image_id": str(uuid4()),
            "scale_factor": 4,
        },
    )

    assert response.status_code == 401


def test_invalid_scale_factor_rejected():
    access_token = register_and_login()

    image_id = upload_test_image(access_token)

    response = client.post(
        "/api/v1/enhancements",
        headers={
            "Authorization": f"Bearer {access_token}",
        },
        json={
            "image_id": image_id,
            "scale_factor": 3,
        },
    )

    assert response.status_code == 400

    assert response.json() == {
        "detail": "Scale factor must be 2 or 4.",
    }


def test_nonexistent_image_rejected():
    access_token = register_and_login()

    image_id = str(uuid4())

    with patch(
        "app.api.v1.enhancements.process_enhancement.delay"
    ) as mock_delay:
        response = client.post(
            "/api/v1/enhancements",
            headers={
                "Authorization": f"Bearer {access_token}",
            },
            json={
                "image_id": image_id,
                "scale_factor": 4,
            },
        )

    assert response.status_code == 404
    assert response.json() == {
        "detail": "Image not found.",
    }

    mock_delay.assert_not_called()


def test_create_enhancement_queues_job():
    access_token = register_and_login()

    image_id = upload_test_image(access_token)

    with patch(
        "app.api.v1.enhancements.process_enhancement.delay"
    ) as mock_delay:
        response = client.post(
            "/api/v1/enhancements",
            headers={
                "Authorization": f"Bearer {access_token}",
            },
            json={
                "image_id": image_id,
                "scale_factor": 4,
            },
        )

    assert response.status_code == 202

    data = response.json()

    assert data["input_image_id"] == image_id
    assert data["result_image_id"] is None
    assert data["scale_factor"] == 4
    assert data["status"] == "QUEUED"
    assert data["error_message"] is None

    mock_delay.assert_called_once_with(data["id"])


def test_get_enhancement_success():
    access_token = register_and_login()

    image_id = upload_test_image(access_token)

    with patch(
        "app.api.v1.enhancements.process_enhancement.delay"
    ) as mock_delay:
        create_response = client.post(
            "/api/v1/enhancements",
            headers={
                "Authorization": f"Bearer {access_token}",
            },
            json={
                "image_id": image_id,
                "scale_factor": 4,
            },
        )

    assert create_response.status_code == 202

    job_id = create_response.json()["id"]

    response = client.get(
        f"/api/v1/enhancements/{job_id}",
        headers={
            "Authorization": f"Bearer {access_token}",
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["id"] == job_id
    assert data["input_image_id"] == image_id
    assert data["scale_factor"] == 4
    assert data["status"] == "QUEUED"

    mock_delay.assert_called_once_with(job_id)


def test_get_enhancement_requires_ownership():
    owner_token = register_and_login(
        email="enhancement-owner@example.com",
    )

    other_token = register_and_login(
        email="enhancement-other@example.com",
    )

    image_id = upload_test_image(owner_token)

    with patch(
        "app.api.v1.enhancements.process_enhancement.delay"
    ):
        create_response = client.post(
            "/api/v1/enhancements",
            headers={
                "Authorization": f"Bearer {owner_token}",
            },
            json={
                "image_id": image_id,
                "scale_factor": 4,
            },
        )

    assert create_response.status_code == 202

    job_id = create_response.json()["id"]

    response = client.get(
        f"/api/v1/enhancements/{job_id}",
        headers={
            "Authorization": f"Bearer {other_token}",
        },
    )

    assert response.status_code == 404

    assert response.json() == {
        "detail": "Enhancement job not found.",
    }


def test_queue_failure_returns_service_unavailable():
    access_token = register_and_login()

    image_id = upload_test_image(access_token)

    with patch(
        "app.api.v1.enhancements.process_enhancement.delay",
        side_effect=RuntimeError("Redis unavailable"),
    ):
        response = client.post(
            "/api/v1/enhancements",
            headers={
                "Authorization": f"Bearer {access_token}",
            },
            json={
                "image_id": image_id,
                "scale_factor": 4,
            },
        )

    assert response.status_code == 503

    assert response.json() == {
        "detail": "Enhancement queue is unavailable.",
    }

    db = TestingSessionLocal()

    try:
        job = (
            db.query(EnhancementJob)
            .order_by(EnhancementJob.created_at.desc())
            .first()
        )

        assert job is not None
        assert job.status == EnhancementJobStatus.FAILED
        assert "Redis unavailable" in job.error_message
    finally:
        db.close()
