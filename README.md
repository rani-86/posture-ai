# PostureAI — AI-Powered Human Posture and Movement Intelligence System

End-to-end computer vision system: video → MediaPipe pose estimation → joint angles →
rule-based posture analysis → ML/DL movement and activity classification → annotated
video report, exposed via a FastAPI backend and a Streamlit dashboard.

**Status: Phases 1-9 complete.** Built incrementally with real validation at each step —
see `docs/phase1_notes.md` through `docs/phase8_notes.md` for the per-phase build log,
and the two documents below for the consolidated picture.

**Start here:**
- [`docs/technical_report.md`](docs/technical_report.md) — pose estimation fundamentals,
  joint-angle math, feature engineering, CNN vs. LSTM, training methodology, real
  evaluation results, and an honest limitations section.
- [`docs/interview_qa.md`](docs/interview_qa.md) — likely interview questions, answered
  from this project's actual decisions and numbers.

## What's real vs. placeholder (read before trusting any number)

| Component | Status |
|---|---|
| Video I/O, MediaPipe pose estimation, smoothing, normalization | Real, tested, runs on any video. Not yet validated against a real human video for detection accuracy (no such video was available in the build environment). |
| Joint angles, posture features, rule-based deviation engine | Real geometry, unit-tested against known configurations. Thresholds are heuristic, documented, and configurable — not clinically validated. |
| Movement classifier (XGBoost/LSTM/GRU, 10 classes) | Real UI-PRMD data, real results (LSTM 0.940 accuracy/0.937 macro-F1). **Not wired into the live API/dashboard** — its feature space (UI-PRMD Kinect angles) doesn't match this project's own MediaPipe pipeline. |
| Activity classifier (sitting/standing/walking/bending/squatting) | Trained **only on synthetic placeholder data** — no real labeled clips were available. Wired into the API/dashboard, but every prediction carries this disclaimer. |
| FastAPI backend, Streamlit dashboard | Real, tested (API via `TestClient` + a live `uvicorn` smoke test; dashboard verified by running the real app — its file-upload widget couldn't be driven by the available browser-automation tool, so that flow needs a quick manual check). |

## Environment

- Python 3.11 (via `py -3.11`; Python 3.13 is also installed on this machine but not
  used here, since `mediapipe`/`xgboost` wheels target 3.11 most reliably on Windows).
- GPU detected: NVIDIA RTX 2050, 4GB VRAM. `torch` is installed CPU-only (the default
  PyPI wheel) — sufficient for this project's small sequence models; GPU install was
  deferred as unnecessary at this scale (see `docs/phase1_notes.md`).

## Setup

```bash
cd posture-ai
py -3.11 -m venv .venv
.venv/Scripts/python.exe -m pip install --only-binary=:all: -r requirements.txt
```

## Running things

```bash
# Train (real UI-PRMD data required - see docs/phase4_notes.md for how it was obtained)
.venv/Scripts/python.exe train.py            # movement classification
.venv/Scripts/python.exe train_activity.py   # activity recognition (synthetic data)

# Evaluate already-trained checkpoints (no retraining)
.venv/Scripts/python.exe evaluate.py

# Run the test suite (81 tests, no large dataset required - uses synthetic fixtures)
.venv/Scripts/python.exe -m pytest -q

# FastAPI backend
.venv/Scripts/python.exe -m uvicorn app.api.main:app --reload
# then: POST /videos, GET /videos/{id}/status, /report, /annotated-video, GET /health

# Streamlit dashboard
.venv/Scripts/python.exe -m streamlit run app/dashboard.py
```

## Dataset decisions (full reasoning in `docs/phase1_notes.md` and `docs/phase4_notes.md`)

- **UI-PRMD** (provided as `Movements.zip`/`Segmented Movements.zip`, Kinect joint-angle
  data): used for real, genuinely-labeled **10-class movement-type classification**.
  Correction made mid-project: the original plan assumed a correct/incorrect
  movement-quality label existed in this data — inspection showed it doesn't, only
  movement type. The task was honestly pivoted rather than fabricating a label.
- **KIMORE**: identified as a possible generalization check in Phase 1, never obtained
  (requires registration) — not used.
- **Activity recognition** (sitting/standing/walking/bending/squatting): no real labeled
  clips were ever obtained. A synthetic, procedurally-generated placeholder dataset
  validates the pipeline instead, with the limitation stated everywhere this model's
  output appears (code, docs, and the dashboard UI itself).

## Project structure

```
posture-ai/
  data/raw/              UI-PRMD extracted data, uploaded videos (gitignored)
  docs/                  phase-by-phase build log + technical_report.md + interview_qa.md
  src/
    data/                 UI-PRMD + synthetic activity dataset loading, torch Dataset
    preprocessing/        video I/O (frame-by-frame, FPS-aware, never loads full video)
    pose_estimation/      MediaPipe wrapper, smoothing, normalization, landmark schema
    feature_extraction/   joint-angle geometry, summary-stat/sequence feature engineering
    posture_analysis/     rule-based posture-deviation engine
    models/                LSTM/GRU architectures
    training/              XGBoost + LSTM/GRU training loops (shared via label_fn)
    evaluation/            metrics, error analysis, pipeline throughput benchmark
    inference/             model registry (singleton-loaded checkpoints), report pipeline
    visualization/         skeleton overlay drawing
    utils/                 config loading, logging
  app/
    api/                   FastAPI backend (Phase 7)
    dashboard.py            Streamlit dashboard (Phase 8)
  configs/config.yaml      all thresholds/hyperparameters/paths - no hardcoded values
  checkpoints/             trained model weights + normalization stats (gitignored)
  reports/                 generated reports/annotated videos (gitignored)
  tests/                   81 tests, synthetic fixtures only (no large dataset needed)
  train.py                 Phase 4: UI-PRMD movement classifier training
  train_activity.py        Phase 5: synthetic activity classifier training
  evaluate.py               Phase 6: checkpoint-based evaluation (no retraining)
```
