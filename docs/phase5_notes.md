# Phase 5 — Movement/Activity Recognition

## Status of the open data item (flagged since Phase 1)

No labeled video clips for sitting/standing/walking/bending/squatting were available in
this environment, and none have been provided. This is unchanged from the Phase 1/4
flag: real clips (self-recorded or sourced) are still needed for genuine activity-
recognition metrics. You chose to proceed by building and validating the full pipeline on
**synthetic, procedurally-generated placeholder data** now, swapping in real clips later.

**Everything reported in this phase's "results" is on synthetic data and describes
pipeline correctness, not real-world accuracy.** This is stated here, in code docstrings
(`src/data/activity_dataset.py`, `train_activity.py`), and in the printed output of
`train_activity.py` itself, so it cannot be mistaken for a real performance claim later.

## What was built

| Module | Responsibility |
|---|---|
| [`src/data/activity_dataset.py`](../src/data/activity_dataset.py) | `generate_synthetic_activity_dataset`: procedurally generates pose-landmark sequences for 5 activity classes via simple forward kinematics (trunk/hip/knee/arm angles → 15 landmark x,y positions), in the **same normalized coordinate frame** (mid-hip origin, torso-length scale) that the real MediaPipe pipeline from Phase 2/3 produces. Each class has characteristic, randomized motion (e.g. squatting = periodic deep knee-flexion cycles; walking = contralateral leg/arm oscillation; bending = a ramp-hold-return trunk-angle profile). `activity_subject_wise_split` mirrors Phase 4's leakage-free split methodology, using synthetic "subject" groups. |
| [`train_activity.py`](../train_activity.py) | End-to-end: generate synthetic data → subject-wise split → train XGBoost/LSTM/GRU → evaluate on held-out synthetic subjects → `reports/phase5_activity_model_comparison.json`. |

**No new model or training code was needed.** Phase 4's `PoseSequenceDataset`,
`train_sequence_model`/`predict_sequence_model`, and `train_xgboost`/`build_feature_matrix`
were generalized (added an optional `label_fn` parameter, default preserves Phase 4's
exact previous behavior — all 55 Phase 1-4 tests still pass unchanged) rather than
duplicated, since the only real difference between Phase 4 and Phase 5's training need is
how to pull a 0-indexed class label out of a sample object.

## Why synthetic data, and why it's a legitimate (if limited) choice here

- Using synthetic data to validate an ML pipeline's mechanics (shapes, training loop,
  evaluation, checkpointing) before real data exists is standard engineering practice -
  the alternative was leaving Phase 5's training code unwritten and untested.
- The synthetic generator deliberately outputs data in the **exact feature space**
  (15-landmark, mid-hip-normalized, 30-channel) that `src/pose_estimation` produces from
  real video — so when real clips are available, only the data loader needs to change
  (MediaPipe pipeline output → `ActivitySample`-shaped sequences), not the models or
  training code. This directly addresses the limitation flagged in Phase 4 (UI-PRMD's
  feature space not matching this project's own pipeline).
- What it cannot validate: whether these classes are actually separable from real human
  movement, whether real pose-estimation noise/occlusion/missing-landmarks (Phase 2's
  hold-forward behavior) degrades accuracy, or whether 5 synthetic "subjects" of
  procedural variation resembles real inter-person variation. None of that is claimed.

## Testing performed

```bash
.venv/Scripts/python.exe -m pytest -q
```
63 tests pass (8 new this phase): generated-dataset shape/finiteness checks, all 5
classes represented, label_fn correctness, a sanity check that standing and squatting
produce measurably different average landmark positions (not just noise), subject-split
disjointness/overlap-rejection, and two reuse-smoke-tests confirming the generalized
Phase 4 training code (`train_sequence_model`, `train_xgboost`) works correctly against
Phase 5's `ActivitySample` objects via `label_fn`.

## Results (synthetic data only — see caveat above)

Full confusion matrices in `reports/phase5_activity_model_comparison.json`.

| Model | Accuracy | Macro Precision | Macro Recall | Macro F1 |
|---|---|---|---|---|
| XGBoost | 1.000 | 1.000 | 1.000 | 1.000 |
| LSTM | 1.000 | 1.000 | 1.000 | 1.000 |
| GRU | 1.000 | 1.000 | 1.000 | 1.000 |

**Perfect accuracy across all three models on the held-out synthetic test subjects.**
This is expected, not impressive: the 5 classes were generated from 5 distinct
parametric templates with only mild Gaussian jitter, so they are far more separable than
real human movement would be — a different person's "sitting" and "bending" overlap far
more than these procedural templates do. A result this clean on real data would be a red
flag for leakage or an overly easy task, not evidence of a strong model. The correct
reading of this number is "the data pipeline, training loop, and evaluation code have no
bugs that would silently produce wrong predictions" — not "this model recognizes
activities with 100% accuracy." Real clips will almost certainly score lower, and that
drop, when it happens, should be read as the dataset becoming realistic, not as a
regression.

## Git commands used this phase

```bash
git add -A
git commit -m "Phase 5: activity recognition pipeline, validated on synthetic placeholder data"
```

## Next steps

- **Blocking for real metrics**: real labeled video clips (self-recorded or sourced) for
  sitting/standing/walking/bending/squatting, still an open item.
- Phase 6 (pending approval): evaluation and error analysis across all models built so
  far (Phase 4 movement classifier + Phase 5 activity classifier), consolidating what's
  real vs. synthetic/placeholder in one place before building the backend/dashboard on
  top of them.
