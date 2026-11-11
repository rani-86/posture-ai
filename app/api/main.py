"""FastAPI backend: video upload, pose/posture/activity processing, report retrieval.

Routes only handle HTTP concerns (validation, status codes, request/response shaping);
all actual work happens in src/inference (model loading) and src/pose_estimation /
src/posture_analysis (the pipelines themselves) - this file should stay thin.
"""
from __future__ import annotations

import uuid
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, HTTPException, UploadFile
from fastapi.responses import FileResponse

from app.api.job_store import job_store
from app.api.schemas import (
    ActivityPredictionResponse,
    HealthResponse,
    JobStatusResponse,
    PostureSummaryResponse,
    ReportResponse,
    UploadResponse,
)
from src.inference.model_registry import get_activity_model
from src.inference.report_pipeline import analyze_video
from src.utils.config import load_config
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)
app = FastAPI(title="PostureAI API", version="0.1.0")

ALLOWED_EXTENSIONS = {".mp4"}
MAX_UPLOAD_BYTES = 200 * 1024 * 1024  # 200 MB


def _process_job(job_id: str, video_path: Path, annotated_path: Path) -> None:
    config = load_config()
    job_store.set_processing(job_id)
    try:
        report = analyze_video(video_path, config, annotated_output_path=annotated_path)
        job_store.set_done(job_id, report, str(annotated_path))
        logger.info(f"Job {job_id} completed")
    except Exception as e:
        logger.exception(f"Job {job_id} failed")
        job_store.set_failed(job_id, str(e))


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    try:
        get_activity_model()
        model_loaded = True
    except FileNotFoundError:
        model_loaded = False
    return HealthResponse(status="ok", activity_model_loaded=model_loaded)


@app.post("/videos", response_model=UploadResponse, status_code=202)
def upload_video(file: UploadFile, background_tasks: BackgroundTasks) -> UploadResponse:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=415, detail=f"Unsupported file type '{suffix}'. Only .mp4 is accepted.")

    config = load_config()
    job_id = str(uuid.uuid4())
    config.raw_data_dir.mkdir(parents=True, exist_ok=True)
    config.reports_dir.mkdir(parents=True, exist_ok=True)

    video_path = config.raw_data_dir / f"{job_id}{suffix}"
    bytes_written = 0
    with open(video_path, "wb") as out:
        while chunk := file.file.read(1024 * 1024):
            bytes_written += len(chunk)
            if bytes_written > MAX_UPLOAD_BYTES:
                out.close()
                video_path.unlink(missing_ok=True)
                raise HTTPException(status_code=413, detail=f"File exceeds {MAX_UPLOAD_BYTES // (1024 * 1024)}MB limit.")
            out.write(chunk)

    if bytes_written == 0:
        video_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    job_store.create(job_id, str(video_path))
    annotated_path = config.reports_dir / f"{job_id}_annotated.mp4"
    background_tasks.add_task(_process_job, job_id, video_path, annotated_path)

    return UploadResponse(job_id=job_id, status="pending", message="Video accepted, processing started.")


@app.get("/videos/{job_id}/status", response_model=JobStatusResponse)
def get_status(job_id: str) -> JobStatusResponse:
    job = job_store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Unknown job_id: {job_id}")
    return JobStatusResponse(job_id=job.job_id, status=job.status, error=job.error)


@app.get("/videos/{job_id}/report", response_model=ReportResponse)
def get_report(job_id: str) -> ReportResponse:
    job = job_store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Unknown job_id: {job_id}")
    if job.status != "done":
        raise HTTPException(status_code=409, detail=f"Job is '{job.status}', report not ready yet.")

    report = job.report
    posture = report.posture_summary
    activity = (
        ActivityPredictionResponse(
            predicted_label=report.activity_prediction.predicted_label,
            confidence=report.activity_prediction.confidence,
            class_probabilities=report.activity_prediction.class_probabilities,
            disclaimer=report.activity_prediction.disclaimer,
        )
        if report.activity_prediction
        else None
    )

    return ReportResponse(
        job_id=job_id,
        posture_summary=PostureSummaryResponse(
            frames_with_pose_detected=posture.frames_with_pose_detected,
            total_frames=posture.total_frames,
            detection_rate=posture.frames_with_pose_detected / posture.total_frames,
            forward_head_deviation_rate=posture.forward_head_deviation_rate,
            shoulder_asymmetry_deviation_rate=posture.shoulder_asymmetry_deviation_rate,
            trunk_inclination_deviation_rate=posture.trunk_inclination_deviation_rate,
            mean_trunk_inclination_deg=posture.mean_trunk_inclination_deg,
        ),
        activity_prediction=activity,
        warnings=report.warnings,
    )


@app.get("/videos/{job_id}/annotated-video")
def get_annotated_video(job_id: str) -> FileResponse:
    job = job_store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Unknown job_id: {job_id}")
    if job.status != "done" or not job.annotated_video_path:
        raise HTTPException(status_code=409, detail=f"Job is '{job.status}', annotated video not ready yet.")
    if not Path(job.annotated_video_path).exists():
        raise HTTPException(status_code=410, detail="Annotated video file no longer exists.")
    return FileResponse(job.annotated_video_path, media_type="video/mp4")
