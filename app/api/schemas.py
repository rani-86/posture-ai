"""Pydantic request/response models for the FastAPI backend - kept separate from both
routes (app/api/routes.py) and inference logic (src/inference), so the API's wire format
can change without touching the pipeline, and vice versa."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

JobStatus = Literal["pending", "processing", "done", "failed"]


class UploadResponse(BaseModel):
    job_id: str
    status: JobStatus
    message: str


class JobStatusResponse(BaseModel):
    job_id: str
    status: JobStatus
    error: str | None = None


class PostureSummaryResponse(BaseModel):
    frames_with_pose_detected: int
    total_frames: int
    detection_rate: float
    forward_head_deviation_rate: float | None
    shoulder_asymmetry_deviation_rate: float | None
    trunk_inclination_deviation_rate: float | None
    mean_trunk_inclination_deg: float | None


class ActivityPredictionResponse(BaseModel):
    predicted_label: str
    confidence: float
    class_probabilities: dict[str, float]
    disclaimer: str


class ReportResponse(BaseModel):
    job_id: str
    posture_summary: PostureSummaryResponse
    activity_prediction: ActivityPredictionResponse | None
    warnings: list[str] = Field(default_factory=list)


class HealthResponse(BaseModel):
    status: Literal["ok"]
    activity_model_loaded: bool


class ErrorResponse(BaseModel):
    detail: str
