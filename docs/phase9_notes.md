# Phase 9 — Testing & Technical Documentation

## What was added

- [`tests/test_report_pipeline.py`](../tests/test_report_pipeline.py): tests
  `analyze_video` (report generation) directly — report structure/types, annotated video
  actually written to disk, correct error on an unreadable video, and the activity
  disclaimer/probability-sum invariants.
- [`tests/test_end_to_end.py`](../tests/test_end_to_end.py): the project's one explicit
  end-to-end test — a short sample video through the *entire* real pipeline (video I/O →
  pose estimation → smoothing/normalization → posture features → deviation assessment →
  activity prediction → annotated video), via the same public entrypoint a real caller
  uses, with the annotated output read back and frame-counted to confirm it's a valid,
  playable video rather than just "a file exists."
- [`docs/technical_report.md`](technical_report.md): pose estimation fundamentals,
  landmark schema, joint-angle math, feature engineering, CNN-vs-LSTM reasoning,
  temporal modeling, training/validation methodology, real evaluation results, inference
  optimization, and a consolidated limitations section.
- [`docs/interview_qa.md`](interview_qa.md): interview Q&A grounded in this project's
  actual decisions and numbers (not generic ML trivia).
- `README.md`: rewritten to reflect the finished state (it had drifted since Phase 1 —
  still described the original, since-corrected UI-PRMD plan).

## Test coverage, mapped to the spec's testing checklist

| Required | Covered by |
|---|---|
| Video loading | `test_video_io.py` |
| Frame extraction | `test_video_io.py` |
| Landmark extraction | `test_pose_estimator.py` |
| Joint-angle calculations | `test_geometry.py`, `test_joint_angles.py` |
| Missing landmarks | `test_temporal_smoothing.py`, `test_posture_features.py` |
| Model input dimensions | `test_sequence_models.py` |
| Prediction output | `test_sequence_models.py`, `test_train_xgboost.py` |
| API endpoints | `test_api.py` |
| Report generation | `test_report_pipeline.py` (new) |
| End-to-end (short sample video) | `test_end_to_end.py` (new) |

```bash
.venv/Scripts/python.exe -m pytest -q
```
**81 tests pass** (5 new this phase). The full suite runs against synthetic fixtures
only — no large/registered dataset is required to run it, by design (see Phase 4's
notes on why UI-PRMD-format fixtures are synthetic in the test suite even though real
UI-PRMD data was used for actual training).

## Engineering standards — self-review against the spec's checklist

- Python 3.11, type hints throughout, clean modular layout: done (Phases 1-8).
- Configurable parameters: thresholds, hyperparameters, and most paths live in
  `configs/config.yaml`, not hardcoded in code. Exception, documented rather than
  silently present: a handful of experiment-specific constants (`TARGET_SEQUENCE_LENGTH`,
  subject-split sets, the UI-PRMD data directory path) live as named module-level
  constants in `train.py`/`train_activity.py` rather than the shared config file — these
  are specific to one training run's experimental design, not deployment-time settings,
  and moving them to `config.yaml` would suggest they're meant to be tuned per-deployment
  when they're actually fixed by the experiment itself.
- Training/inference separation: `train.py`/`train_activity.py` (train) vs. `evaluate.py`
  (inference-only, loads checkpoints) vs. `src/inference` (used by the API/dashboard).
- Reproducibility: fixed seeds (`config.seed`), subject-wise splits, checkpoint +
  normalization-stats persistence verified by `evaluate.py` reproducing identical metrics.
- Logging and exception handling: shared `loguru` logger throughout; API errors return
  clean HTTP responses without leaking stack traces (logged server-side via
  `logger.exception`).
- CPU and GPU support: `config.device: auto`, falls back to CPU; GPU install
  deliberately deferred (documented, not silently skipped).

## Git commands used this phase

```bash
git add -A
git commit -m "Phase 9: testing consolidation, technical report, interview Q&A"
```

## What's left (beyond this project's current scope)

- Real human video to validate pose-estimation detection accuracy.
- Real labeled activity clips to retrain the Phase 5 model on genuine data.
- UI-PRMD's official joint-documentation, to enable a real Kinect-vs-Vicon MPJPE metric.
- KIMORE registration, if the generalization check from Phase 1's plan is still wanted.

These are the same open items flagged throughout the project — restated here as the
honest "definition of done" rather than left implicit.
