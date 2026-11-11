"""In-memory job tracking for uploaded-video processing.

Deliberately simple (a dict behind a lock, not a database or task queue) - scoped to
what this project needs: a single-process API that can report job status and results.
This does not survive a process restart and does not scale across multiple workers;
documented as a limitation in docs/phase7_notes.md rather than quietly assumed.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field

from app.api.schemas import JobStatus
from src.inference.report_pipeline import VideoAnalysisReport


@dataclass
class Job:
    job_id: str
    status: JobStatus = "pending"
    video_path: str = ""
    annotated_video_path: str | None = None
    report: VideoAnalysisReport | None = None
    error: str | None = None


class JobStore:
    def __init__(self):
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def create(self, job_id: str, video_path: str) -> Job:
        job = Job(job_id=job_id, video_path=video_path)
        with self._lock:
            self._jobs[job_id] = job
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def set_processing(self, job_id: str) -> None:
        with self._lock:
            self._jobs[job_id].status = "processing"

    def set_done(self, job_id: str, report: VideoAnalysisReport, annotated_video_path: str | None) -> None:
        with self._lock:
            job = self._jobs[job_id]
            job.status = "done"
            job.report = report
            job.annotated_video_path = annotated_video_path

    def set_failed(self, job_id: str, error: str) -> None:
        with self._lock:
            job = self._jobs[job_id]
            job.status = "failed"
            job.error = error


job_store = JobStore()
