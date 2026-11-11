# Phase 2 — Video Preprocessing & Pose Estimation

## What was built

| Module | Responsibility |
|---|---|
| [`src/preprocessing/video_io.py`](../src/preprocessing/video_io.py) | `get_video_metadata`, frame-by-frame `iter_frames` generator (never loads the full video into memory), FPS-aware downsampling that preserves original timestamps, `AnnotatedVideoWriter` for incremental output. |
| [`src/pose_estimation/landmark_schema.py`](../src/pose_estimation/landmark_schema.py) | The 15-landmark subset this project uses (nose, ears, shoulders, elbows, wrists, hips, knees, ankles), mapped to MediaPipe's 33-point indices; `Landmark`/`PoseFrame` dataclasses; skeleton bone list. |
| [`src/pose_estimation/pose_estimator.py`](../src/pose_estimation/pose_estimator.py) | Wraps `mediapipe.solutions.pose.Pose`; applies `landmark_confidence_threshold` per-landmark (visibility below threshold → landmark marked absent, not silently kept). |
| [`src/pose_estimation/temporal_smoothing.py`](../src/pose_estimation/temporal_smoothing.py) | Moving-average smoothing per landmark over `smoothing_window` frames; missing landmarks are held at their last known value (visibility forced to 0 so downstream code can tell "held" apart from "observed"), not zero-filled or interpolated. |
| [`src/pose_estimation/normalization.py`](../src/pose_estimation/normalization.py) | Centers on mid-hip, rescales by torso length (shoulder-midpoint to hip-midpoint distance), so posture features aren't confounded by camera distance or frame size. Does not correct for viewing angle — documented limitation. |
| [`src/visualization/skeleton.py`](../src/visualization/skeleton.py) | Draws bones/joints over a frame; joints held-forward by the smoother are colored differently from freshly observed ones. |
| [`src/pose_estimation/pipeline.py`](../src/pose_estimation/pipeline.py) | Wires the above into one frame-by-frame pipeline: `run_pose_pipeline(video_path, config, annotated_output_path=None)` yields `(normalized_frame, raw_frame)` per processed frame and optionally writes an annotated video. |
| [`src/utils/config.py`](../src/utils/config.py) | Typed loader for `configs/config.yaml` (no hardcoded paths/thresholds in code). |

## Key decisions

- **Frame-by-frame generators, not frame lists**, throughout — `iter_frames` and
  `run_pose_pipeline` are generators; a long video is never materialized in memory.
- **Downsampling preserves original timestamps.** `iter_frames(path, target_fps=15)` on a
  30fps source keeps every other frame and reports `timestamp_sec = source_index /
  source_fps`, not a timestamp recomputed at 15fps — so timestamps stay aligned with the
  source video regardless of the processing rate chosen.
- **Missing landmarks are held forward, not interpolated or zeroed.** Zero-filling would
  inject a fake (0,0) coordinate into later joint-angle math; interpolation needs
  look-ahead that breaks frame-by-frame streaming. Holding forward is the simplest
  defensible choice for short gaps; documented as a limitation for long occlusions.
- **Confidence filtering happens per-landmark, not per-frame** — a frame can have some
  landmarks present and others absent (e.g. an arm out of frame while the torso is
  visible), which matters for the trunk/head posture features in Phase 3.

## Testing performed

```bash
.venv/Scripts/python.exe -m pytest -q
```
15 tests pass, covering: video metadata extraction, FPS downsampling + timestamp
preservation, missing-file handling, temporal smoothing math, missing-landmark
hold-forward behavior, coordinate normalization (centering/scaling and passthrough when
reference points are missing), skeleton drawing (non-mutating, handles fully-missing
frames), and pose-estimator confidence filtering (via a mocked MediaPipe result, since a
solid-color synthetic frame cannot exercise real person detection).

**End-to-end smoke test**: ran `run_pose_pipeline` on a synthetic 30-frame video through
the full pipeline (pose estimation → smoothing → normalization → skeleton overlay →
annotated video write) with no crashes. Detection rate was 0% as expected — the synthetic
video contains no actual person, so this validates the plumbing (frame counts, generator
behavior, video writer), not detection accuracy.

## What is NOT yet validated

Real pose-detection accuracy on an actual human video has not been tested — this requires
a real MP4 of a person, which Phase 2 did not have access to in this environment. This
should be the first thing checked once a sample video is available: run
`run_pose_pipeline` on it and visually inspect `reports/*_annotated.mp4` for correct
skeleton tracking before relying on Phase 3's joint-angle math downstream.

## Git commands used this phase

```bash
git add -A
git commit -m "Phase 2: video I/O, MediaPipe pose estimation, smoothing, normalization, skeleton overlay"
```

## Next (Phase 3, pending approval)

Joint-angle calculation (law-of-cosines on the normalized landmarks) and the rule-based
posture feature extraction (head alignment, shoulder symmetry, trunk inclination, etc.)
on top of the `PoseFrame` stream this phase produces.
