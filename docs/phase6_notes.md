# Phase 6 — Evaluation & Error Analysis

## What was built

| Module | Responsibility |
|---|---|
| [`evaluate.py`](../evaluate.py) | Inference-only entrypoint: loads saved checkpoints (does NOT retrain) for both Phase 4 and Phase 5 models, re-evaluates on the same held-out subjects, runs error analysis, benchmarks the Phase 2 pipeline, writes `reports/phase6_evaluation_report.json`. |
| [`src/evaluation/error_analysis.py`](../src/evaluation/error_analysis.py) | `top_confused_pairs`: largest off-diagonal confusion-matrix entries. `pairwise_class_separability`: a model-independent check of whether two classes are inherently close in feature space (normalized mean-difference), to distinguish "the model failed to learn a real distinction" from "these classes are genuinely similar." |
| [`src/evaluation/pipeline_benchmark.py`](../src/evaluation/pipeline_benchmark.py) | `benchmark_pipeline`: wall-clock time, FPS, and peak Python-heap memory for the full Phase 2 pose-estimation pipeline. |

Also fixed a real gap found while building this: `train.py`/`train_activity.py` were
saving model weights but not the per-channel normalization statistics computed during
training, which are required to correctly preprocess data at inference time. Both now
also save `checkpoints/*_normalize_stats.npz`; this was necessary for `evaluate.py` to
do genuine checkpoint-based inference rather than silently recomputing (and leaking)
statistics from whatever data it's evaluating on.

## Results — UI-PRMD movement classification (reloaded from checkpoint, not retrained)

Re-running the Phase 4 checkpoints through `evaluate.py` reproduced **identical** metrics
to the original training run (XGBoost 0.930/0.928, LSTM 0.940/0.937, GRU 0.900/0.897
accuracy/macro-F1) — confirming the save/load round-trip (model weights + normalization
stats) is correct, not just that training once produced a number.

### Error analysis: why m07 and m10 get confused

All three models confuse movement class `m07` with `m10` far more than any other pair
(10, 10, and 11 of 20 test examples respectively). Phase 4's notes flagged this as
"independent of architecture, so more likely genuine exercise similarity than a
model-specific weakness" — this phase checked that claim quantitatively rather than
leaving it as a guess:

```
separability(m07, m10)              = 0.426   (normalized mean-feature distance)
separability(m01, m02), never confused = 0.801
```

m07 and m10 are **about half as separable** in the 330-dimensional summary-statistic
feature space as a typical never-confused pair. This is real, data-grounded evidence for
the hypothesis, not just a repeated confusion-matrix pattern — these two exercises
produce genuinely similar joint-angle signatures in this feature representation, and no
amount of additional training on this same data/feature space is likely to resolve it
without either more discriminative features or knowing (from UI-PRMD's documentation,
not available here) whether these two exercises are anatomically similar by design.

## Results — synthetic activity recognition (reloaded from checkpoint)

All three models reproduce the Phase 5 result exactly: 1.000 accuracy/macro-F1. As
stated in Phase 5, this is confirmation the pipeline has no bugs, not a real-world
accuracy claim — restated here because this is the kind of number that's easy to
misquote later without the caveat attached.

## Pose-estimation accuracy (PCK/MPJPE) — deliberately NOT computed, and why

The master prompt asks for landmark metrics (PCK/MPJPE) "where ground-truth keypoints
are available." The provided UI-PRMD zip does contain both Kinect and Vicon position
data for the same recordings, which looked like a promising real ground-truth source
(Vicon marker-based motion capture vs. Kinect depth-camera skeleton). On inspection:

- Kinect: 22 joints × 3 coordinates, ~30fps (765 frames in a sample file).
- Vicon: **39** markers × 3 coordinates, ~90-100fps (2300 frames, same recording).

Computing MPJPE requires a verified joint-to-joint correspondence between the two
skeletons (which of Vicon's 39 markers corresponds to which of Kinect's 22 joints) plus
temporal resampling and a coordinate-frame alignment (the two systems don't share origin,
axis convention, or units — Vicon values are in the hundreds/thousands, Kinect's in the
tens). UI-PRMD's official documentation presumably defines this correspondence, but it
was not included in the provided zip files. Guessing a joint mapping between a 22-point
and a 39-point skeleton without that documentation would produce a number that looks
like a real metric but isn't trustworthy — exactly the kind of fabricated result this
project is explicit about avoiding. **This metric is deferred, not computed with an
unverified assumption.** It could be revisited if UI-PRMD's documentation PDF becomes
available, or once a real video with manually-annotated ground-truth keypoints exists
(the more direct way to validate the actual MediaPipe pipeline this project uses, as
opposed to a different depth-camera system).

## Video-processing performance (Phase 2 pipeline)

Benchmarked on a 150-frame synthetic video (320×240 @ 15fps) through the full pipeline
(decode → MediaPipe Pose → smoothing → normalization):

| Metric | Value |
|---|---|
| Frames processed | 150 |
| Wall-clock time | 5.37 s |
| Throughput | 28.0 fps |
| Peak Python-heap memory | 28.2 MB |

**Caveats, stated plainly:**
- Measured on CPU-only torch/MediaPipe on this machine (RTX 2050 present but not used by
  MediaPipe's CPU-based XNNPACK delegate) - a GPU-accelerated MediaPipe build would likely
  be faster; not attempted this phase.
- `peak_memory_mb` is measured via Python's `tracemalloc`, which tracks Python-level heap
  allocations only - it does **not** count memory held by MediaPipe/OpenCV's native (C++)
  buffers, which are the actual majority of a real pose-estimation pipeline's memory use.
  This number is a meaningful lower bound and a regression-detection signal (did Python-
  side memory usage change unexpectedly), not a measurement of total process RSS.
- Measured on a synthetic (no-person) video - per-frame compute cost should be
  representative of a real video (MediaPipe runs its full detection pass regardless of
  outcome), but this hasn't been confirmed against a real clip.

## Testing performed

```bash
.venv/Scripts/python.exe -m pytest -q
```
68 tests pass (5 new this phase): confused-pair extraction and top-N behavior, class
separability on identical vs. well-separated synthetic distributions, and a benchmark
smoke test confirming `benchmark_pipeline` returns sane values on a synthetic video.

## Git commands used this phase

```bash
git add -A
git commit -m "Phase 6: consolidated evaluation, error analysis, pipeline benchmark"
```

## Next (Phase 7, pending approval)

FastAPI backend wrapping the inference pipelines built so far (video upload, pose
estimation, posture analysis, movement/activity classification, report generation).
