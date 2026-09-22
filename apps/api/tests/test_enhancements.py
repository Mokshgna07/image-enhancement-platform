from __future__ import annotations

import io
from unittest.mock import patch
from uuid import UUID, uuid4
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


def get_error_body(response):
    body = response.json()

    assert body["success"] is False
    assert "error" in body
    assert "code" in body["error"]
    assert "message" in body["error"]
    assert body["request_id"]

    return body


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


def upload_test_image(
    access_token: str,
    filename: str = "test.jpg",
) -> str:
    response = client.post(
        "/api/v1/images/upload",
        headers={
            "Authorization": f"Bearer {access_token}",
        },
        files={
            "file": (
                filename,
                create_test_jpeg(),
                "image/jpeg",
            )
        },
    )

    assert response.status_code == 201

    return response.json()["id"]


def create_enhancement(
    access_token: str,
    image_id: str,
) -> dict:
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
    mock_delay.assert_called_once_with(response.json()["id"])

    return response.json()


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

    body = get_error_body(response)

    assert body["error"]["code"] == "bad_request"
    assert body["error"]["message"] == "Scale factor must be 2 or 4."


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

    body = get_error_body(response)

    assert body["error"]["code"] == "resource_not_found"
    assert body["error"]["message"] == "Image not found."

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
    assert data["error"] is None

    mock_delay.assert_called_once_with(data["id"])


def test_get_enhancement_success():
    access_token = register_and_login()

    image_id = upload_test_image(access_token)

    data = create_enhancement(
        access_token,
        image_id,
    )

    job_id = data["id"]

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
    assert data["error"] is None

    assert "error_message" not in data
    assert "storage_path" not in data


def test_get_enhancement_requires_ownership():
    owner_token = register_and_login(
        email="enhancement-owner@example.com",
    )

    other_token = register_and_login(
        email="enhancement-other@example.com",
    )

    image_id = upload_test_image(owner_token)

    data = create_enhancement(
        owner_token,
        image_id,
    )

    job_id = data["id"]

    response = client.get(
        f"/api/v1/enhancements/{job_id}",
        headers={
            "Authorization": f"Bearer {other_token}",
        },
    )

    assert response.status_code == 404

    body = get_error_body(response)

    assert body["error"]["code"] == "resource_not_found"
    assert body["error"]["message"] == "Enhancement job not found."


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

    body = get_error_body(response)

    assert body["error"]["code"] == "service_unavailable"
    assert body["error"]["message"] == "Enhancement queue is unavailable."

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


def test_list_images_pagination_and_total():
    access_token = register_and_login(
        email="image-history@example.com",
    )

    image_ids = [
        upload_test_image(access_token, filename=f"image-{index}.jpg")
        for index in range(3)
    ]

    response = client.get(
        "/api/v1/images",
        headers={
            "Authorization": f"Bearer {access_token}",
        },
        params={
            "page": 1,
            "page_size": 2,
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["page"] == 1
    assert data["page_size"] == 2
    assert data["total"] == 3
    assert len(data["items"]) == 2

    returned_ids = [item["id"] for item in data["items"]]

    assert set(returned_ids).issubset(set(image_ids))

    for item in data["items"]:
        assert "storage_path" not in item
        assert item["original_filename"].startswith("image-")


def test_list_images_second_page():
    access_token = register_and_login(
        email="image-page-two@example.com",
    )

    for index in range(3):
        upload_test_image(
            access_token,
            filename=f"page-{index}.jpg",
        )

    response = client.get(
        "/api/v1/images",
        headers={
            "Authorization": f"Bearer {access_token}",
        },
        params={
            "page": 2,
            "page_size": 2,
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["page"] == 2
    assert data["page_size"] == 2
    assert data["total"] == 3
    assert len(data["items"]) == 1


def test_list_images_user_isolation():
    owner_token = register_and_login(
        email="image-owner@example.com",
    )

    other_token = register_and_login(
        email="image-other@example.com",
    )

    owner_image_id = upload_test_image(
        owner_token,
        filename="owner.jpg",
    )

    upload_test_image(
        other_token,
        filename="other.jpg",
    )

    response = client.get(
        "/api/v1/images",
        headers={
            "Authorization": f"Bearer {owner_token}",
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["total"] == 1
    assert len(data["items"]) == 1
    assert data["items"][0]["id"] == owner_image_id
    assert data["items"][0]["original_filename"] == "owner.jpg"


def test_list_images_pagination_validation():
    access_token = register_and_login(
        email="image-validation@example.com",
    )

    invalid_page = client.get(
        "/api/v1/images",
        headers={
            "Authorization": f"Bearer {access_token}",
        },
        params={
            "page": 0,
        },
    )

    assert invalid_page.status_code == 422

    invalid_page_size = client.get(
        "/api/v1/images",
        headers={
            "Authorization": f"Bearer {access_token}",
        },
        params={
            "page_size": 101,
        },
    )

    assert invalid_page_size.status_code == 422


def test_get_image_requires_ownership():
    owner_token = register_and_login(
        email="single-image-owner@example.com",
    )

    other_token = register_and_login(
        email="single-image-other@example.com",
    )

    image_id = upload_test_image(
        owner_token,
        filename="private.jpg",
    )

    response = client.get(
        f"/api/v1/images/{image_id}",
        headers={
            "Authorization": f"Bearer {other_token}",
        },
    )

    assert response.status_code == 404

    body = get_error_body(response)

    assert body["error"]["code"] == "resource_not_found"
    assert body["error"]["message"] == "Image not found."


def test_list_enhancements_pagination_and_total():
    access_token = register_and_login(
        email="enhancement-history@example.com",
    )

    image_ids = [
        upload_test_image(
            access_token,
            filename=f"enhancement-{index}.jpg",
        )
        for index in range(3)
    ]

    jobs = [
        create_enhancement(
            access_token,
            image_id,
        )
        for image_id in image_ids
    ]

    response = client.get(
        "/api/v1/enhancements",
        headers={
            "Authorization": f"Bearer {access_token}",
        },
        params={
            "page": 1,
            "page_size": 2,
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["page"] == 1
    assert data["page_size"] == 2
    assert data["total"] == 3
    assert len(data["items"]) == 2

    returned_ids = [item["id"] for item in data["items"]]

    assert set(returned_ids).issubset(
        {job["id"] for job in jobs}
    )

    for item in data["items"]:
        assert "error_message" not in item
        assert item["error"] is None
        assert item["processing_metadata"] is None


def test_list_enhancements_second_page():
    access_token = register_and_login(
        email="enhancement-page-two@example.com",
    )

    for index in range(3):
        image_id = upload_test_image(
            access_token,
            filename=f"job-page-{index}.jpg",
        )

        create_enhancement(
            access_token,
            image_id,
        )

    response = client.get(
        "/api/v1/enhancements",
        headers={
            "Authorization": f"Bearer {access_token}",
        },
        params={
            "page": 2,
            "page_size": 2,
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["page"] == 2
    assert data["page_size"] == 2
    assert data["total"] == 3
    assert len(data["items"]) == 1


def test_list_enhancements_user_isolation():
    owner_token = register_and_login(
        email="job-owner@example.com",
    )

    other_token = register_and_login(
        email="job-other@example.com",
    )

    owner_image_id = upload_test_image(
        owner_token,
        filename="owner-job.jpg",
    )

    other_image_id = upload_test_image(
        other_token,
        filename="other-job.jpg",
    )

    owner_job = create_enhancement(
        owner_token,
        owner_image_id,
    )

    create_enhancement(
        other_token,
        other_image_id,
    )

    response = client.get(
        "/api/v1/enhancements",
        headers={
            "Authorization": f"Bearer {owner_token}",
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["total"] == 1
    assert len(data["items"]) == 1
    assert data["items"][0]["id"] == owner_job["id"]
    assert data["items"][0]["input_image_id"] == owner_image_id


def test_list_enhancements_pagination_validation():
    access_token = register_and_login(
        email="enhancement-validation@example.com",
    )

    invalid_page = client.get(
        "/api/v1/enhancements",
        headers={
            "Authorization": f"Bearer {access_token}",
        },
        params={
            "page": 0,
        },
    )

    assert invalid_page.status_code == 422

    invalid_page_size = client.get(
        "/api/v1/enhancements",
        headers={
            "Authorization": f"Bearer {access_token}",
        },
        params={
            "page_size": 101,
        },
    )

    assert invalid_page_size.status_code == 422
def test_cancel_queued_enhancement():
    access_token = register_and_login(
        email="cancel-queued@example.com",
    )

    image_id = upload_test_image(access_token)

    job = create_enhancement(
        access_token,
        image_id,
    )

    job_id = job["id"]

    response = client.post(
        f"/api/v1/enhancements/{job_id}/cancel",
        headers={
            "Authorization": f"Bearer {access_token}",
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["id"] == job_id
    assert data["status"] == "CANCELED"
    assert data["result_image_id"] is None
    assert data["error"] is None


def test_cancel_enhancement_requires_ownership():
    owner_token = register_and_login(
        email="cancel-owner@example.com",
    )

    other_token = register_and_login(
        email="cancel-other@example.com",
    )

    image_id = upload_test_image(owner_token)

    job = create_enhancement(
        owner_token,
        image_id,
    )

    response = client.post(
        f"/api/v1/enhancements/{job['id']}/cancel",
        headers={
            "Authorization": f"Bearer {other_token}",
        },
    )

    assert response.status_code == 404

    body = get_error_body(response)

    assert body["error"]["code"] == "resource_not_found"
    assert body["error"]["message"] == "Enhancement job not found."


def test_cancel_nonexistent_enhancement():
    access_token = register_and_login(
        email="cancel-missing@example.com",
    )

    job_id = str(uuid4())

    response = client.post(
        f"/api/v1/enhancements/{job_id}/cancel",
        headers={
            "Authorization": f"Bearer {access_token}",
        },
    )

    assert response.status_code == 404

    body = get_error_body(response)

    assert body["error"]["code"] == "resource_not_found"
    assert body["error"]["message"] == "Enhancement job not found."


def test_cancel_processing_enhancement_rejected():
    access_token = register_and_login(
        email="cancel-processing@example.com",
    )

    image_id = upload_test_image(access_token)

    job_data = create_enhancement(
        access_token,
        image_id,
    )

    db = TestingSessionLocal()

    try:
        job = db.get(
            EnhancementJob,
            UUID(job_data["id"]),
        )

        assert job is not None

        job.status = EnhancementJobStatus.PROCESSING
        db.commit()
    finally:
        db.close()

    response = client.post(
        f"/api/v1/enhancements/{job_data['id']}/cancel",
        headers={
            "Authorization": f"Bearer {access_token}",
        },
    )

    assert response.status_code == 409

    body = get_error_body(response)

    assert body["error"]["code"] == "conflict"
    assert (
        body["error"]["message"]
        == "Only queued enhancement jobs can be canceled. "
        "Current status: PROCESSING."
    )


def test_cancel_completed_enhancement_rejected():
    access_token = register_and_login(
        email="cancel-completed@example.com",
    )

    image_id = upload_test_image(access_token)

    job_data = create_enhancement(
        access_token,
        image_id,
    )

    db = TestingSessionLocal()

    try:
        job = db.get(
            EnhancementJob,
	    UUID(job_data["id"]),
        )

        assert job is not None

        job.status = EnhancementJobStatus.COMPLETED
        db.commit()
    finally:
        db.close()

    response = client.post(
        f"/api/v1/enhancements/{job_data['id']}/cancel",
        headers={
            "Authorization": f"Bearer {access_token}",
        },
    )

    assert response.status_code == 409

    body = get_error_body(response)

    assert body["error"]["code"] == "conflict"
    assert (
        body["error"]["message"]
        == "Only queued enhancement jobs can be canceled. "
        "Current status: COMPLETED."
    )


def test_cancel_already_canceled_enhancement_rejected():
    access_token = register_and_login(
        email="cancel-again@example.com",
    )

    image_id = upload_test_image(access_token)

    job_data = create_enhancement(
        access_token,
        image_id,
    )

    db = TestingSessionLocal()

    try:
        job = db.get(
            EnhancementJob,
            UUID(job_data["id"]),
        )

        assert job is not None

        job.status = EnhancementJobStatus.CANCELED
        db.commit()
    finally:
        db.close()

    response = client.post(
        f"/api/v1/enhancements/{job_data['id']}/cancel",
        headers={
            "Authorization": f"Bearer {access_token}",
        },
    )

    assert response.status_code == 409

    body = get_error_body(response)

    assert body["error"]["code"] == "conflict"
    assert (
        body["error"]["message"]
        == "Only queued enhancement jobs can be canceled. "
        "Current status: CANCELED."
    )
def test_worker_skips_canceled_enhancement():
    access_token = register_and_login(
        email="worker-canceled@example.com",
    )

    image_id = upload_test_image(access_token)

    job_data = create_enhancement(
        access_token,
        image_id,
    )

    db = TestingSessionLocal()

    try:
        job = db.get(
            EnhancementJob,
            UUID(job_data["id"]),
        )

        assert job is not None

        job.status = EnhancementJobStatus.CANCELED
        db.commit()
    finally:
        db.close()

    with patch(
        "app.workers.tasks.SessionLocal",
        TestingSessionLocal,
    ):
        with patch(
            "app.workers.tasks.EnhancementTask.inference_service",
        ) as mock_inference:
            from app.workers.tasks import process_enhancement

            process_enhancement.run(job_data["id"])

            mock_inference.enhance.assert_not_called()

    db = TestingSessionLocal()

    try:
        job = db.get(
            EnhancementJob,
            UUID(job_data["id"]),
        )

        assert job is not None
        assert job.status == EnhancementJobStatus.CANCELED
        assert job.started_at is None
        assert job.completed_at is None
        assert job.failed_at is None
    finally:
        db.close()
