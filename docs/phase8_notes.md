# Phase 8 — Streamlit Dashboard

## What was built

[`app/dashboard.py`](../app/dashboard.py) — a single-page Streamlit app with the 9
sections from the project spec:

1. Project overview — states plainly which components are real (pose estimation,
   joint angles, posture rules, movement classifier) vs. placeholder (activity
   classifier trained on synthetic data), linking to the phase docs.
2. Video upload (`st.file_uploader`, `.mp4` only).
3. Video preview (`st.video` of the raw upload).
4. (Processing happens on a "Run analysis" button between sections 3 and 5.)
5. Pose estimation visualization — the skeleton-overlay annotated video.
6. Posture analysis — forward-head/shoulder/trunk deviation rates as metrics.
7. Joint-angle analytics — line chart of knee angles, trunk inclination, forward-head
   angle over time.
8. Temporal movement timeline — area chart of detection/deviation per frame.
9. Activity recognition — predicted label, confidence, per-class probability bar chart,
   with the synthetic-data disclaimer shown as a visible warning, not buried in text.
10. Detailed analysis report — raw JSON report in an expander, plus a CSV download of
    the full frame-level data.

(The numbering above is 1-10 because "processing" sits between upload/preview and the
result sections, rather than being its own numbered page — the spec's 9 sections map
onto these naturally.)

## Key decision: dashboard calls `src/inference` directly, not the FastAPI backend

Documented in the module docstring: a Streamlit app is already a single Python process,
so routing through its own sibling FastAPI service (Phase 7) would add an HTTP round
trip and a second server to keep running, for no benefit at this project's scale. The
FastAPI backend remains the integration point for a separate client; the dashboard is a
second, independent consumer of the same `src/inference.report_pipeline.analyze_video`
function Phase 7 uses — not a wrapper around the API.

## Why the report pipeline grew a frame-level field this phase

`VideoAnalysisReport` (Phase 7) was aggregate-only by design, to keep the API response
small. The dashboard's joint-angle/timeline charts need per-frame data, so
`src/inference/report_pipeline.py` gained `FrameRecord` and `VideoAnalysisReport.
frame_records` — additive, not a breaking change to Phase 7's API (the FastAPI schema
in `app/api/schemas.py` was deliberately left untouched; the API response still doesn't
include frame-level data).

## Testing performed, and an honest limitation

```bash
.venv/Scripts/python.exe -m pytest -q
```
76 tests pass (1 new this phase): `test_dashboard.py` verifies the data-transformation
layer (`_frame_records_df`) produces the expected shape/columns from a real
`analyze_video` run — this is the part of the dashboard that's meaningfully unit-testable.

**The actual Streamlit UI (widgets, layout, button clicks) was verified manually, not
via automated test**, for two reasons: Streamlit's widget tree isn't something pytest
exercises meaningfully, and the available browser-automation tool in this environment
could not drive Streamlit's file-upload dropzone (no exposed `<input type="file">`
element, and no file-upload capability in the available toolset). What WAS verified:

- Started the real app (`streamlit run app/dashboard.py`) and loaded it in a browser -
  confirmed the overview section renders with the correct real/placeholder disclaimers,
  the file uploader appears, and there are no import/startup errors (which would have
  surfaced immediately, e.g. the `src.inference` imports failing if run from the wrong
  working directory).
- The upload → analyze → chart flow itself is exercised indirectly: it's the exact same
  `analyze_video` call path already proven by Phase 7's API tests and live uvicorn smoke
  test (same function, same inputs/outputs) - only the Streamlit rendering on top of
  that is unverified by automation.

**Recommended manual check before considering this phase fully done**: run the app
locally and actually upload a video through the browser to confirm the charts render as
expected - this is explicitly flagged rather than silently assumed.

## Git commands used this phase

```bash
git add -A
git commit -m "Phase 8: Streamlit dashboard across 9 sections, calling src/inference directly"
```

## Next (Phase 9, pending approval)

Testing consolidation and the full technical documentation deliverable (pose estimation
fundamentals, joint-angle math, CNN vs. LSTM, limitations, interview Q&A).
