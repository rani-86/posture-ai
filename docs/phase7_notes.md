# Phase 7 — FastAPI Backend

## What was built

| Module | Responsibility |
|---|---|
| [`app/api/main.py`](../app/api/main.py) | FastAPI app: `POST /videos` (upload + validate + kick off background processing), `GET /videos/{id}/status`, `GET /videos/{id}/report`, `GET /videos/{id}/annotated-video`, `GET /health`. Routes only handle HTTP concerns — all real work delegates to `src/inference`. |
| [`app/api/schemas.py`](../app/api/schemas.py) | Pydantic request/response models, kept separate from both routes and inference logic. |
| [`app/api/job_store.py`](../app/api/job_store.py) | In-memory, lock-protected job tracking (`pending`/`processing`/`done`/`failed`). |
| [`src/inference/model_registry.py`](../src/inference/model_registry.py) | Loads the Phase 5 activity-recognition checkpoint once per process (`lru_cache`), not per request. |
| [`src/inference/report_pipeline.py`](../src/inference/report_pipeline.py) | `analyze_video`: runs the Phase 2 pose pipeline + Phase 3 posture analysis + Phase 5 activity prediction on a real video, producing one structured report. |
| [`src/feature_extraction/pose_channels.py`](../src/feature_extraction/pose_channels.py) | Flattens a `PoseFrame` into the exact 30-channel layout the activity models expect — the piece that makes Phase 5's models usable on real pipeline output. |

## Key decision: which trained models actually get wired into the API

Only the **Phase 5 activity classifier** is loaded for live inference. The **Phase 4
UI-PRMD movement classifier is deliberately not exposed**: it was trained on UI-PRMD's
66-channel Kinect Euler-angle feature space, which is not the same feature space this
project's own MediaPipe pipeline produces (flagged as a limitation since Phase 4).
Wiring it into the API would mean feeding it 30 MediaPipe channels where it expects 66
UI-PRMD channels — either a hard crash or, worse, a shape-compatible but meaningless
input if someone padded/truncated to fit. Neither is acceptable, so it's left out with
the reasoning documented here rather than silently worked around.

The activity classifier IS wired in, with its own caveat carried end-to-end: every
`/report` response includes `activity_prediction.disclaimer`, stating plainly that the
model was trained only on synthetic placeholder data (Phase 5). No API consumer can get
this prediction without the caveat attached in the same response.

## Design choices

- **Processing is asynchronous** (`BackgroundTasks`): upload returns immediately with a
  `job_id` and `pending` status; a client polls `/status` then fetches `/report`. This
  matches the master prompt's requirement to process frame-by-frame without blocking,
  and avoids a long-lived HTTP request timing out on a longer video.
- **Validation**: rejects non-`.mp4` uploads (415), empty files (400), and files over
  200MB (413, checked incrementally while streaming to disk rather than loading the
  whole upload into memory first).
- **Structured logging**: all modules use the shared `loguru`-based logger from
  `src/utils/logging_utils.py`; job failures are logged with full tracebacks
  (`logger.exception`) before being surfaced to the client as a generic error message
  (so stack traces don't leak into API responses).
- **In-memory job store**: a deliberate, documented limitation — no database, no task
  queue. Fine for this project's scope (a single-process demo/portfolio backend); would
  need replacing (Redis/Postgres + Celery or similar) before any real multi-worker or
  persistent deployment. Not implemented, to avoid speculative infrastructure for a
  requirement nobody asked for.

## Testing performed

```bash
.venv/Scripts/python.exe -m pytest -q
```
75 tests pass (7 new this phase, via `fastapi.testclient.TestClient`): health check,
upload rejection for wrong file type and empty file, 404 on unknown job IDs, 409 on
fetching a report for a non-done job, and a full upload → background-process →
status → report lifecycle test on a synthetic video.

**Live smoke test** (not part of the automated suite — run manually to validate real
HTTP behavior, not just the in-process `TestClient`):
```bash
.venv/Scripts/python.exe -m uvicorn app.api.main:app --host 127.0.0.1 --port 8123
curl http://127.0.0.1:8123/health
curl -X POST http://127.0.0.1:8123/videos -F "file=@<video>.mp4;type=video/mp4"
curl http://127.0.0.1:8123/videos/<job_id>/status
curl http://127.0.0.1:8123/videos/<job_id>/report
curl -o annotated.mp4 http://127.0.0.1:8123/videos/<job_id>/annotated-video
```
Confirmed: `/health` reports the activity model loaded; a real upload is accepted,
processed, and produces a report with the activity disclaimer attached and a correct
low-detection warning (synthetic video, no real person); the annotated video downloads
as a valid non-empty MP4 file.

## Git commands used this phase

```bash
git add -A
git commit -m "Phase 7: FastAPI backend - upload, async processing, report, health check"
```

## Next (Phase 8, pending approval)

Streamlit dashboard: project overview, video upload, pose visualization, posture
analysis, joint-angle analytics, activity recognition, timeline, detailed report — built
on top of this API (or directly on the inference modules, to be decided in Phase 8).
