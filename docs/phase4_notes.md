# Phase 4 — Dataset Preparation & ML Model Training

## Correction to the Phase 1 plan (read this first)

You provided the real UI-PRMD release as `Movements.zip` and `Segmented Movements.zip`.
Inspecting the actual files surfaced an error in the Phase 1 plan: **this base release
does not include a correct/incorrect movement-quality label** — only Kinect/Vicon
joint-angle and joint-position time series, organized by **movement type** (`m01`-`m10`),
**subject** (`s01`-`s10`), and **repetition** (`e01`-`e10`). There is no per-repetition
"this one is a deviant/incorrect example" field anywhere in the files.

Rather than invent a correct/incorrect label that is not in the data (which the master
prompt explicitly rules out — "do not fabricate labels"), Phase 4 implements the task
this data actually supports: **10-class movement-type classification** from joint-angle
sequences. This is still a genuine supervised learning problem with real, dataset-provided
ground truth (the label is literally which of the 10 instructed exercises the subject was
performing), and it still supports a meaningful XGBoost-vs-LSTM-vs-GRU comparison. The
originally-planned "posture-quality" classification task is deferred — it would need
either a different UI-PRMD-derived release that includes deviation labels, or a separate
clinician-labeled dataset, neither of which is in the files provided.

KIMORE (continuous clinical quality scores) remains a plausible source for a true
movement-quality regression/classification task later, but was not provided this phase
and still requires registration — unchanged from Phase 1.

## Data used

- **Source**: `Segmented Movements/Kinect/Angles/*.txt` from the provided UI-PRMD zip —
  1000 files (10 movements × 10 subjects × 10 repetitions, no gaps), each a CSV of
  66 columns (22 Kinect skeleton joints × 3 Euler angles per joint) × T frames
  (T ranges 39–160, median 71).
- Kinect (not Vicon) was chosen deliberately: Vicon is high-end marker-based motion
  capture, which is a poor match for this project's consumer-camera framing; Kinect's
  coarser, single-sensor skeleton is closer in spirit to what MediaPipe-from-a-webcam
  produces, even though the two are not numerically interchangeable.
- The 66 columns are used in their original file order as the model's feature space
  without asserting a specific joint-name mapping. UI-PRMD's official documentation
  defines that mapping, but it was not redistributed with the data provided here, and
  hard-coding an unverified guess would misrepresent precision this project doesn't
  actually have. This only affects explainability (you get "channel 14 drove this
  prediction" rather than "left knee drove this prediction"), not classification validity.
- **Split: subject-wise**, not episode-wise — train = subjects 1-7, val = subject 8,
  test = subjects 9-10. A repetition-wise random split would leak each subject's
  individual movement style across train and test, inflating test accuracy relative to
  how the model would perform on an unseen person.

## What was built

| Module | Responsibility |
|---|---|
| [`src/data/uiprmd_dataset.py`](../src/data/uiprmd_dataset.py) | Parses UI-PRMD filenames into `Episode(movement_id, subject_id, episode_id, sequence)`; `subject_wise_split` enforcing disjoint subjects across train/val/test. |
| [`src/feature_extraction/sequence_features.py`](../src/feature_extraction/sequence_features.py) | `summary_statistics_features` (mean/std/min/max/range per channel → fixed vector for XGBoost); `pad_or_truncate_sequence` (fixed-length sequence for the RNNs, padding by repeating the last frame rather than zero-filling, to avoid implying a sudden snap-to-origin motion). |
| [`src/data/sequence_dataset.py`](../src/data/sequence_dataset.py) | `PoseSequenceDataset`: torch `Dataset` wrapper; per-channel z-score normalization fit only on the training split and passed explicitly into val/test construction, so no evaluation-set statistics leak into feature scaling. |
| [`src/models/sequence_models.py`](../src/models/sequence_models.py) | `LSTMClassifier`, `GRUClassifier`: recurrent layer(s) → final hidden state → linear classifier head. |
| [`src/training/train_xgboost.py`](../src/training/train_xgboost.py) | Multi-class XGBoost on summary-statistic features, early-stopped on the validation split. |
| [`src/training/train_sequence_model.py`](../src/training/train_sequence_model.py) | Shared train loop for LSTM/GRU: Adam + cross-entropy, early stopping on val loss, best-checkpoint restoration. |
| [`src/evaluation/classification_metrics.py`](../src/evaluation/classification_metrics.py) | `evaluate_predictions`: accuracy, macro precision/recall/F1, per-class F1, confusion matrix — identical scoring applied to all three models for a fair comparison. |
| [`train.py`](../train.py) | End-to-end: load → subject-wise split → train XGBoost/LSTM/GRU → evaluate all three on the same held-out test subjects → write `reports/phase4_model_comparison.json`. |

## Testing performed

```bash
.venv/Scripts/python.exe -m pytest -q
```
55 tests pass (19 new this phase) using small **synthetic** fixture files in UI-PRMD's
exact format — not the real downloaded data — so the test suite doesn't depend on a
dataset that requires registration/manual download to be present. This covers: filename
parsing, malformed-file/wrong-shape error handling, subject-wise split disjointness,
summary-statistic feature math (verified against hand-computed values), padding/
truncation behavior, LSTM/GRU forward-pass output shapes, a full train+predict smoke
test, XGBoost feature-matrix shape and fit/predict, and classification-metric correctness
against known perfect/wrong-prediction cases.

**Real-data run**: `train.py` was additionally run against the actual extracted UI-PRMD
files (not a test, a real training run) — see the results below.

## Results (real UI-PRMD data, test subjects 9-10, 10-class movement classification)

Full confusion matrices in `reports/phase4_model_comparison.json`.

| Model | Accuracy | Macro Precision | Macro Recall | Macro F1 |
|---|---|---|---|---|
| XGBoost | 0.930 | 0.951 | 0.930 | 0.928 |
| **LSTM** | **0.940** | **0.953** | **0.940** | **0.937** |
| GRU | 0.900 | 0.913 | 0.900 | 0.897 |

LSTM edges out both XGBoost and GRU here. With only 7 training subjects this is not a
statistically strong claim about LSTM "being better" in general — it's plausible that a
different subject split (different 2 held out for test) would change the ranking. What's
more informative than the headline ranking is the **error pattern, consistent across all
three models**: movement class 7 (`m07`) is confused with class 10 (`m10`) in every model
(10/20, 10/20, and 9/20 of class-7 test examples respectively misclassified as class 10).
Because this confusion shows up independent of model architecture, it's more likely to
reflect genuine similarity between those two exercises in the 66-channel Kinect angle
space (e.g. two shoulder-rotation-type movements with overlapping joint-angle ranges) than
a weakness specific to one model. Confirming exactly which two exercises `m07`/`m10` are
would need UI-PRMD's official movement-list documentation, which wasn't provided with the
data — noted here rather than guessed.

## Honest limitations

- 10 subjects total, 2 held out for test — small by deep-learning standards; test-set
  metrics have real variance from which 2 subjects happened to be held out, not just from
  model quality. A different subject split would likely shift numbers by several points.
- All subjects are healthy adults performing instructed exercises — the model has not
  seen clinical populations, injuries, or non-standard movement styles. It classifies
  "which of these 10 prescribed exercises is this," not "is this movement healthy."
- The 66-channel feature space is UI-PRMD/Kinect-specific Euler-angle output. It is NOT
  the same feature space as the MediaPipe landmarks from Phase 2/3 — these models cannot
  be applied directly to this project's own webcam pose-estimation pipeline without a
  separate feature-mapping or retraining step. This is a real, load-bearing limitation
  for Phase 5+ and will need to be addressed (either by extracting MediaPipe-equivalent
  joint angles from UI-PRMD's 3D positions instead of using its pre-computed Euler
  angles, or by training a separate model on self-collected MediaPipe data for Phase 5's
  activity-recognition task, which was already the plan since UI-PRMD doesn't cover
  sitting/standing/walking/bending anyway).

## Git commands used this phase

```bash
git add -A
git commit -m "Phase 4: UI-PRMD ingestion, XGBoost vs LSTM vs GRU movement classification"
```

## Next (Phase 5, pending approval)

Movement/activity recognition (sitting/standing/walking/bending/squatting) on MediaPipe
landmark sequences — the self-curated dataset flagged as an open item since Phase 1.
