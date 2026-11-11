# Phase 1 — Environment Setup & Dataset Research

## Environment

| Item | Choice | Why |
|---|---|---|
| Python | 3.11.4 (`py -3.11`) | Machine also has 3.13 as default; `mediapipe`, `xgboost` wheels and the broader CV/ML ecosystem are most reliably prebuilt for 3.11 on Windows. Spec also asked for 3.11. |
| Virtual env | `posture-ai/.venv` | Isolated from other projects in this workspace. |
| GPU | NVIDIA RTX 2050, 4GB VRAM, driver CUDA 13.3 | Enough for LSTM/GRU and XGBoost; not enough for heavy video backbones — training configs will use small batch sizes and CPU-friendly defaults, with CUDA used opportunistically (`project.device: auto` in `configs/config.yaml`). |
| Pose backend | MediaPipe Pose, pinned `0.10.14` | MediaPipe `>=1.0` removed the legacy `solutions.*` API in favor of a Tasks API with a different interface; pinning avoids a breaking API change mid-project. Verified `mediapipe.solutions.pose.Pose(...)` loads under 0.10.14. |

All dependency versions are pinned in `requirements.txt` and the venv was built strictly
from that file (not from ad-hoc `pip install`s) to avoid the numpy 2.x / pandas 3.x ABI
drift that an unpinned install pulled in during setup (torch 2.3.1 expects numpy<2).

## Dataset research and decision

The project touches three distinct analytical tasks that should not share one dataset
dishonestly:

### 1. Geometric posture indicators — no dataset needed
Head alignment, shoulder symmetry, trunk inclination, knee angle etc. are computed
directly from landmark geometry (law of cosines on 2D/3D joint coordinates). This is
deterministic and rule-based, not learned — evaluated by correctness of the math and by
sanity-checking against visually-labeled frames, not by a training set.

### 2. Supervised posture-quality / movement-deviation classification
**Primary dataset: UI-PRMD** (University of Idaho Physical Rehabilitation Movement Data).
- Vicon-captured and Kinect-captured 3D joint angle/position time series for 10
  physical-therapy exercises (e.g. deep squat, hurdle step, shoulder abduction), each
  performed correctly and with clinician-informed incorrect variants.
- Fits directly: already in joint-angle/landmark-sequence space, genuine per-repetition
  correct/incorrect labels, suitable for both XGBoost-on-engineered-features and
  LSTM/GRU-on-raw-sequence approaches, enabling an honest model comparison (Phase 4).
- Documented limitation: healthy-subject population only, lab motion-capture rather than
  consumer RGB video, and exercise-specific rather than general everyday posture — so
  the trained model's claim is "this repetition deviates from the reference movement
  pattern for this exercise," not "this person has bad posture in general."

**Secondary / generalization check: KIMORE**.
- RGB + depth + skeleton + continuous clinical quality scores, healthy and pathological
  subjects (Parkinson's, stroke, low back pain).
- Used only to evaluate whether a UI-PRMD-trained model's quality signal transfers to a
  different population and labeling rubric — not for primary training, given its smaller
  size and different scoring scale (continuous score vs. UI-PRMD's correct/incorrect).

### 3. Activity recognition (sitting / standing / walking / bending / squatting)
No existing public skeleton-labeled dataset lines up with exactly these five classes at
a size this hardware can train in reasonable time (NTU RGB+D has no squat class; forcing
an unrelated benchmark's labels onto this task would misrepresent what was measured).

Decision: build a small, explicitly self-curated dataset — MediaPipe-extracted pose
landmark sequences from short permissively-licensed / public-domain clips and
short self-recorded webcam clips, manually labeled into the 5 classes. This will be
reported in the technical document as a custom, limited-size dataset with real (and
likely modest) metrics — never presented as a benchmark result.

### What this buys us (and what it doesn't)
- Real labels, real training, real evaluation metrics for two separate supervised tasks.
- A rule-based layer that is explainable and doesn't depend on any dataset.
- Explicit boundaries: posture-quality claims are scoped to the exercises UI-PRMD covers;
  activity-recognition claims are scoped to the self-curated dataset's size and diversity.
  Both limitations will be restated in the final technical report (Phase 9).

## Proposed architecture (summary, detail in Phase 2–9 docs as built)

```
Video → frame sampling (target_fps) → MediaPipe Pose per frame → landmark confidence
filter → temporal smoothing → joint-angle feature extraction
   ├─► rule-based posture_analysis engine (deterministic, configurable thresholds)
   ├─► XGBoost on per-window engineered features (posture-quality classification)
   └─► LSTM/GRU on raw landmark sequences (posture-quality AND activity recognition)
→ visualization (skeleton overlay, angle annotations, timeline) → report
FastAPI backend wraps inference; Streamlit dashboard wraps FastAPI + visualization.
```

## Verification performed this phase

```bash
py -3.11 -m venv .venv
.venv/Scripts/python.exe -m pip install --only-binary=:all: -r requirements.txt
.venv/Scripts/python.exe -c "import torch, mediapipe, xgboost, fastapi, streamlit, numpy, pandas, cv2, sklearn, matplotlib; mediapipe.solutions.pose.Pose(static_image_mode=True)"
```

Two pin issues surfaced and were fixed during setup:
- `matplotlib==3.9.1` has no Windows cp311 wheel (only `3.9.1.post1`/`3.9.2`) — pinned to
  `3.9.2` in `requirements.txt`.
- An initial unpinned `pip install mediapipe` resolved to `mediapipe 1.2.2`, which dropped
  the legacy `solutions.*` API in favor of a Tasks API — reinstalled strictly from the
  pinned `requirements.txt` (`mediapipe==0.10.14`) to get the API this project targets.
- `--only-binary=:all:` was used throughout to avoid source builds silently falling back
  to a broken local mingw/x86 toolchain on this machine.

Final confirmed, importable versions in `posture-ai/.venv`:

| Package | Version |
|---|---|
| torch | 2.3.1+cpu (CPU-only build; no CUDA-enabled wheel installed yet — see note below) |
| torchvision | 0.18.1+cpu |
| mediapipe | 0.10.14 (`solutions.pose.Pose` verified working) |
| xgboost | 2.1.0 |
| numpy | 1.26.4 |
| pandas | 2.2.2 |
| opencv-python | 4.10.0.84 (reports underlying OpenCV core `4.11.0` via `cv2.__version__`) |
| matplotlib | 3.9.2 |
| scikit-learn | 1.5.1 |
| fastapi | 0.112.0 |
| streamlit | 1.37.1 |

**GPU note:** `pip install torch==2.3.1` from the default PyPI index installs the CPU-only
build; `torch.cuda.is_available()` is currently `False` despite the RTX 2050 being present.
Getting CUDA-enabled torch requires installing from PyTorch's CUDA-specific index
(`--index-url https://download.pytorch.org/whl/cu121` or similar) instead of plain PyPI.
Deferred to Phase 4/5 (model training) since XGBoost and a small LSTM/GRU on short pose
sequences train acceptably on CPU; GPU install will be revisited only if training time
becomes a real bottleneck.

## Open items before Phase 2
- Need to actually download UI-PRMD and KIMORE (both require a short registration/request
  form from the original research groups — this will need the user's email/affiliation
  to request access, or confirmation to proceed with publicly mirrored subsets if the
  user prefers not to register).
- Need 10–20 short clips per activity class for the self-curated dataset — can be
  self-recorded via webcam in Phase 2/5, or sourced from public-domain clips the user
  points us to.
