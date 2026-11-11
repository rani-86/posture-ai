# PostureAI — AI-Powered Human Posture and Movement Intelligence System

End-to-end computer vision system: video → pose estimation → joint angles → posture
features → ML/DL classification (posture quality + activity) → annotated video report.

Status: **Phase 1 — environment setup & dataset research** (see `docs/phase1_notes.md`
once added; summary below). Built incrementally, phase by phase — see project root chat
history for approvals.

## Environment

- Python 3.11 (via `py -3.11` on this machine; Python 3.13 is also installed but not used
  here because `mediapipe`/`xgboost` wheels target 3.11 most reliably on Windows).
- GPU detected: NVIDIA RTX 2050, 4GB VRAM (CUDA 13.3 driver). Sufficient for the LSTM/GRU
  and XGBoost components; not sized for large video backbones — training is designed to
  run on CPU or this small GPU, batch sizes kept modest accordingly.

## Dataset decision (Phase 1)

Two distinct ML tasks are in scope, and they do not share one dataset honestly:

1. **Movement-quality / posture-deviation classification** (is this repetition of an
   exercise/posture "correct" or "deviant", and which joint angles drive that).
   → **UI-PRMD** (University of Idaho Physical Rehabilitation Movement Data).
   - Vicon + Kinect 3D joint angle/position sequences for 10 physical-therapy exercises,
     performed by healthy subjects, with correct and clinician-informed incorrect
     variants per repetition.
   - Chosen because it ships genuine sequence-level labels for correctness, already in
     joint-angle space — a direct fit for LSTM/GRU temporal modeling and for XGBoost on
     hand-engineered per-repetition features. No label fabrication needed.
   - Limitation to document: healthy-subject-only, Kinect/Vicon skeletons rather than
     RGB video, exercise-specific (not general standing/sitting posture) — so this model
     answers "is this movement pattern deviant", not "is this a bad real-world posture".

2. **Secondary / cross-check dataset — KIMORE** (Kinect Movement Rehabilitation).
   - RGB + depth + skeleton + continuous clinical quality scores (healthy and pathological
     subjects: Parkinson's, stroke, low back pain). Used as an evaluation/comparison set
     for the posture-quality model to check generalization beyond UI-PRMD's healthy-only
     population — not for primary training, given its smaller size and scoring-rubric
     differences.

3. **Activity recognition (sitting/standing/walking/bending/squatting)** has no single
   well-labeled public dataset that matches these exact five classes in skeleton form at
   a size trainable on this hardware. Rather than force-fit NTU RGB+D (no squat class) or
   fabricate performance on an unrelated benchmark, Phase 1 plan is to build a small,
   explicitly self-curated dataset: MediaPipe-extracted pose sequences from short
   permissively-licensed/public-domain clips and/or short self-recorded webcam clips,
   manually labeled per the 5 classes. This will be clearly documented in the technical
   report as a custom, limited-size dataset — real labels, real training, real (modest)
   metrics — not a benchmark claim.

This keeps the three analytical layers honestly separated, as required:
- **Geometric posture indicators** (deterministic, rule-based: trunk inclination, head
  forward position, shoulder symmetry) — no ML, fully explainable, configurable thresholds.
- **Supervised posture-quality classification** (XGBoost on engineered joint-angle
  features + LSTM/GRU on raw landmark sequences) — trained on UI-PRMD, evaluated also on
  KIMORE.
- **Activity recognition** (LSTM/GRU over pose-landmark sequences) — trained on the
  self-curated 5-class dataset, reported with real confusion matrices/F1, caveated as
  small-sample.

## Project structure

```
posture-ai/
  data/                 raw/, processed/ (gitignored)
  notebooks/            exploratory analysis
  src/
    data/                dataset loading & splits
    preprocessing/       video decode, frame sampling, normalization
    pose_estimation/     MediaPipe/YOLO-Pose wrappers, landmark extraction
    feature_extraction/  joint angles, posture geometric features
    posture_analysis/    rule-based posture deviation engine
    models/              XGBoost wrapper, LSTM/GRU architectures
    training/            training loops, experiment config
    evaluation/          metrics (accuracy/F1/confusion matrix/MPJPE/PCK), error analysis
    inference/           model loading + prediction pipeline
    visualization/       skeleton overlay, timelines, report rendering
    utils/               logging, config loading, common types
  app/
    api/                 FastAPI routes
    (streamlit dashboard added in Phase 8)
  configs/               config.yaml (thresholds, hyperparameters, paths)
  checkpoints/           trained model weights (gitignored)
  reports/               generated analysis reports/annotated videos (gitignored)
  tests/
  train.py / evaluate.py / predict.py   (added from Phase 4 onward)
```

## Setup

```bash
cd posture-ai
py -3.11 -m venv .venv
source .venv/Scripts/activate
pip install -r requirements.txt
```
