# PostureAI — Technical Report

This document explains the concepts, math, and design decisions behind PostureAI, and is
written to be read on its own (not just as a changelog of phases). For the detailed
per-phase build log and real results, see `docs/phase1_notes.md` through
`docs/phase8_notes.md`.

## 1. Pose estimation fundamentals

Human pose estimation locates anatomical keypoints (joints) in an image or video frame.
Two broad families exist:

- **Top-down**: detect the person first (a bounding box), then estimate keypoints within
  that crop. More accurate per-person, slower with many people.
- **Bottom-up**: detect all keypoints in the image first, then group them into people.
  Faster with many people, can struggle with keypoint-to-person association in crowds.

This project uses **MediaPipe Pose** (via its `solutions.pose` API, pinned to version
0.10.14 since later releases replaced this API with a different Tasks interface). It is
a single-person, top-down-style model optimized for real-time CPU inference, which is
the right trade-off for a webcam-facing application: this project never needs multi-person
tracking, and real-time CPU performance matters more than squeezing out a percentage
point of accuracy a heavier model might offer.

MediaPipe Pose outputs 33 landmarks per detected person, each with:
- `x`, `y`: normalized image coordinates in [0, 1] (fraction of frame width/height).
- `z`: a depth estimate, relative to the hips, in the same normalized scale as `x`.
- `visibility`: a confidence-like score in [0, 1] for whether the landmark is likely
  visible and correctly located (not occluded, not off-screen).

## 2. Human body landmarks used in this project

Of MediaPipe's 33 landmarks, this project uses a 15-point subset relevant to posture and
movement analysis (`src/pose_estimation/landmark_schema.py`):

> nose, left/right ear, left/right shoulder, left/right elbow, left/right wrist,
> left/right hip, left/right knee, left/right ankle

Keeping an explicit named subset (rather than passing all 33 through every module) makes
feature extraction and visualization code easier to reason about and keeps the project's
missing-landmark and confidence-filtering logic scoped to what's actually used.

**Why not use `z` (depth) in the angle math?** A single RGB camera has no real depth
sensor; MediaPipe's `z` is inferred, not measured, and is materially noisier than `x`/`y`
in practice. Using it in angle calculations would add noise rather than accuracy, so all
geometric features in this project (`src/feature_extraction/geometry.py`) are 2D,
image-plane projections. This is a real, stated limitation (see Section 10), not an
oversight.

## 3. Joint-angle mathematics

Given three points `A`, `B`, `C`, the angle at vertex `B` (e.g. the knee, with `A`=hip,
`B`=knee, `C`=ankle) is computed via the **law of cosines**, equivalently the angle
between vectors `B→A` and `B→C`:

```
cos(angle) = (BA · BC) / (|BA| * |BC|)
angle = arccos(cos(angle))
```

This gives an angle in [0°, 180°]: 180° means the joint is fully extended (hip-knee-ankle
in a straight line), smaller values mean more flexion (bend). The same formula computes
elbow angle (shoulder-elbow-wrist) and hip angle (shoulder-hip-knee).

For **tilt/lean** features (shoulder level, trunk inclination), a different
computation is used: the angle between a body-segment vector (e.g. hip-midpoint →
shoulder-midpoint, for trunk) and a reference axis (vertical or horizontal), folded into
[0°, 90°] since "leaning" is symmetric left-right — a 10° lean to the left and a 10° lean
to the right are the same severity of deviation, even though the raw vector angle would
differ by sign.

**Forward-head angle** approximates the clinical "craniovertebral angle" (CVA): the angle
between a horizontal line through the shoulder and the shoulder→ear line. In real
physiotherapy assessment, CVA < ~50° commonly indicates forward head posture — this
project's default threshold (`forward_head_angle_deg: 50`) mirrors that heuristic. Crucial
caveat: CVA is only meaningful from a **side (sagittal) view** of the subject; from a
front-facing camera, the shoulder-to-ear horizontal offset it depends on nearly vanishes.

## 4. Feature engineering

Two different feature representations are used for two different model families
(`src/feature_extraction/sequence_features.py`):

- **Summary-statistic features** (for XGBoost, which needs fixed-size tabular input):
  per-channel mean, std, min, max, and range across all frames of a sequence. This
  collapses a variable-length time series into one fixed-length vector per sample,
  losing temporal order but capturing the overall range and variability of each channel.
- **Fixed-length sequences** (for LSTM/GRU, which consume temporal order directly):
  padded (by repeating the last frame, not zero-filling, to avoid implying a sudden
  motion) or truncated to a target length.

Both are preceded by **per-channel z-score normalization**, with the mean/std computed
only on the training split and reused (not recomputed) for validation/test — avoiding
the common mistake of letting evaluation-set statistics leak into feature scaling.

## 5. CNN vs. LSTM — why this project uses LSTM/GRU, not a CNN, for sequence classification

A **CNN** (convolutional neural network) is well-suited to data with strong *local
spatial* structure — nearby pixels in an image are related, and convolution exploits
that with shared, spatially-local filters. A **1D CNN over time** can also work for
sequences, by treating nearby *time steps* as "local," using dilated/strided
convolutions to build up a longer receptive field.

An **LSTM/GRU** (recurrent network) instead maintains an explicit hidden state updated
one time step at a time, which is a more natural fit when:
- Sequence length varies a lot between samples (true here: UI-PRMD episodes range
  39-160 frames) — an RNN handles variable length natively; a CNN needs fixed-size
  input or more careful padding/masking.
- The task benefits from an explicit notion of "what has happened so far," not just
  locally-correlated patterns — a squat's deep-knee-bend phase is only recognizable in
  the context of the stand-bend-stand cycle around it.

This project trains and compares **both LSTM and GRU** (not a CNN) because the task —
classifying a whole movement/activity from its joint-angle time series — is naturally
sequential, and because the master prompt specifically calls for temporal sequence
modeling via LSTM/GRU as a distinct skill from CNN-based spatial feature learning. A 1D
CNN was not tried this project; it would be a reasonable additional baseline but wasn't
necessary to answer the project's actual question (does a temporal model beat a
non-temporal one — see Section 8).

## 6. Temporal sequence modeling in this project

Both `LSTMClassifier` and `GRUClassifier` (`src/models/sequence_models.py`) share the
same shape: a recurrent layer (1-2 layers, `num_layers`/`hidden_size` from
`configs/config.yaml`) processes the (T, C) input sequence; the **final hidden state**
(not an average or max over time) feeds a linear classifier head. Using the final
hidden state, rather than pooling over all time steps, means the model must carry
forward everything relevant to the prediction through its recurrence — a deliberate,
standard choice for "classify the whole sequence" tasks (as opposed to per-timestep
labeling, which would pool or output at every step).

LSTM adds a separate cell state and three gates (input/forget/output) to GRU's single
hidden state and two gates (update/reset) — LSTM has more parameters and can in
principle model longer-range dependencies, GRU is cheaper and sometimes matches or beats
LSTM on shorter sequences. This project's own results (Section 8) show LSTM slightly
ahead of GRU on the real UI-PRMD task, consistent with (but not strong evidence for,
given the small dataset) that general pattern.

## 7. Training and validation methodology

- **Subject-wise splitting, not sample-wise**: for both UI-PRMD (Phase 4) and the
  synthetic activity dataset (Phase 5), train/val/test splits are made by *subject*, so
  no subject's data appears in more than one split. A sample-wise random split would let
  the model see one subject's movement style in training and be tested on *more
  examples from that same subject* — inflating test accuracy relative to how the model
  would perform on a genuinely new person.
- **Early stopping on validation loss**, not training loss, with the best checkpoint
  (by val loss) restored at the end — prevents reporting a late-epoch, overfit state as
  the final model.
- **Normalization statistics fit on train only**: the same leakage principle applied to
  per-channel z-score normalization, not just the split itself.

## 8. Model evaluation — real results

**UI-PRMD movement classification** (10 classes, real labeled data, held-out subjects
9-10):

| Model | Accuracy | Macro F1 |
|---|---|---|
| XGBoost | 0.930 | 0.928 |
| LSTM | 0.940 | 0.937 |
| GRU | 0.900 | 0.897 |

All three models share the same confusion (movement `m07` ↔ `m10`), and a direct
feature-space separability check (Phase 6) confirmed this is genuine similarity between
those two exercises, not an architecture-specific weakness: separability 0.43 between
m07/m10 vs. 0.80 for a typical never-confused pair.

**Synthetic activity recognition**: all three models score 1.000 accuracy — explicitly
*not* a real-world claim (Section 10) — a sign the pipeline has no bugs, not that
activity recognition is "solved."

See `docs/phase4_notes.md`, `docs/phase5_notes.md`, and `docs/phase6_notes.md` for full
confusion matrices and per-class breakdowns.

## 9. Inference optimization

- **Models load once per process** (`src/inference/model_registry.py`, `lru_cache`), not
  per request — a checkpoint load is comparatively expensive (deserializing weights);
  doing it once at first use and reusing the in-memory model is the standard pattern.
- **Streaming video processing**: `src/preprocessing/video_io.py`'s `iter_frames` is a
  generator, never loading a full video into memory — `src/pose_estimation/pipeline.py`
  processes and discards each frame before reading the next.
- **Benchmarked throughput** (Phase 6, synthetic 320×240 video, CPU-only): ~28 fps,
  28MB peak Python-heap memory (native MediaPipe/OpenCV buffers not included in that
  number — see Phase 6 notes for the full caveat on `tracemalloc`'s blind spot).
- **CPU vs. GPU**: this project's torch install is CPU-only (the default PyPI wheel);
  MediaPipe's CPU XNNPACK delegate is used regardless. GPU acceleration was deferred as
  unnecessary at this project's scale (small sequence models, modest video resolutions)
  — documented in `docs/phase1_notes.md`, not silently skipped.

## 10. Computer vision limitations (read this before trusting any number in this project)

- **2D projection, camera-angle dependent**: every angle/posture feature is a 2D
  projection from one camera viewpoint. Forward-head angle needs a side view; trunk/
  shoulder tilt need a front view; no single camera placement supports every feature
  well (Phase 3).
- **No clinically-validated thresholds**: posture deviation thresholds
  (`configs/config.yaml`) are heuristic starting points from common physiotherapy
  rules of thumb, kept configurable precisely because a single fixed cutoff does not
  generalize across body types, camera angles, or contexts.
- **Feature-space fragmentation across models**: UI-PRMD's Kinect Euler-angle features
  (66 channels) and this project's own MediaPipe landmark features (30 channels) are
  different spaces. The real, well-validated UI-PRMD movement classifier cannot be
  applied to this project's own video pipeline output without retraining or a verified
  feature mapping — and is deliberately not exposed in the API/dashboard for that reason
  (Phase 7).
- **Activity recognition trained only on synthetic data**: no real labeled video clips
  for sitting/standing/walking/bending/squatting were available. The 1.000 accuracy
  reported for this model describes pipeline correctness on deliberately-separable
  procedural data, not real-world classification ability (Phase 5).
- **No PCK/MPJPE pose-estimation accuracy metric**: the provided UI-PRMD data has both
  Kinect (22 joints) and Vicon (39 markers) for the same recordings, which looked like a
  ground-truth opportunity, but no verified joint-correspondence mapping between the two
  skeletons was available, and guessing one would produce an unverifiable number
  (Phase 6) — deferred rather than fabricated.
- **Missing-landmark handling is a simplification**: Phase 2 holds a missing landmark at
  its last known value for a short window, then (in the API/dashboard layer) falls back
  to 0.0 if never observed — reasonable for brief occlusion, not a substitute for
  genuinely re-detecting a lost limb.
