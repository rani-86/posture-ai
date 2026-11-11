import io

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.api.main import app

client = TestClient(app)


def _make_video_bytes(n_frames: int = 20, fps: int = 10, size: tuple[int, int] = (64, 48)) -> bytes:
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "test.mp4"
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
        for i in range(n_frames):
            writer.write(np.full((size[1], size[0], 3), fill_value=i % 256, dtype=np.uint8))
        writer.release()
        return path.read_bytes()


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert isinstance(body["activity_model_loaded"], bool)


def test_upload_rejects_non_mp4():
    response = client.post("/videos", files={"file": ("video.avi", io.BytesIO(b"not a real video"), "video/avi")})
    assert response.status_code == 415


def test_upload_rejects_empty_file():
    response = client.post("/videos", files={"file": ("video.mp4", io.BytesIO(b""), "video/mp4")})
    assert response.status_code == 400


def test_status_unknown_job_returns_404():
    response = client.get("/videos/does-not-exist/status")
    assert response.status_code == 404


def test_report_unknown_job_returns_404():
    response = client.get("/videos/does-not-exist/report")
    assert response.status_code == 404


def test_upload_and_full_lifecycle_on_synthetic_video():
    video_bytes = _make_video_bytes()
    upload_response = client.post(
        "/videos", files={"file": ("sample.mp4", io.BytesIO(video_bytes), "video/mp4")}
    )
    assert upload_response.status_code == 202
    job_id = upload_response.json()["job_id"]

    # TestClient runs BackgroundTasks synchronously before returning, so processing
    # should already be done by the time we check status.
    status_response = client.get(f"/videos/{job_id}/status")
    assert status_response.status_code == 200
    assert status_response.json()["status"] == "done"

    report_response = client.get(f"/videos/{job_id}/report")
    assert report_response.status_code == 200
    report = report_response.json()
    assert report["posture_summary"]["total_frames"] == 20
    # No real person in the synthetic video, so pose detection is expected to fail -
    # this just confirms the report structure and warning plumbing work correctly.
    assert report["posture_summary"]["detection_rate"] == 0.0
    assert any("only 0%" in w for w in report["warnings"])


def test_report_not_ready_for_pending_job(monkeypatch):
    from app.api.job_store import job_store

    job_store.create("fake-pending-job", "unused.mp4")
    response = client.get("/videos/fake-pending-job/report")
    assert response.status_code == 409
