# Phase 3 — Joint-Angle Calculation & Posture Feature Extraction

## What was built

| Module | Responsibility |
|---|---|
| [`src/feature_extraction/geometry.py`](../src/feature_extraction/geometry.py) | Pure 2D geometry primitives: `angle_at_vertex` (law of cosines, for classic joint angles), `angle_from_vertical` / `angle_from_horizontal` (for tilt/lean features), `midpoint`, `vector`. |
| [`src/feature_extraction/joint_angles.py`](../src/feature_extraction/joint_angles.py) | `compute_joint_angles`: elbow (shoulder-elbow-wrist), knee (hip-knee-ankle), hip (shoulder-hip-knee) angles, 180°=extended. Returns `None` per-angle when a required landmark is absent, rather than guessing. |
| [`src/posture_analysis/posture_features.py`](../src/posture_analysis/posture_features.py) | Posture-specific features: approximate forward-head angle (craniovertebral-angle-style), shoulder tilt, hip tilt, trunk inclination (hip→shoulder vs. vertical), body inclination (ankle→shoulder vs. vertical). |
| [`src/posture_analysis/posture_engine.py`](../src/posture_analysis/posture_engine.py) | `assess_posture`: compares `PostureFeatures` against `configs/config.yaml`'s `posture_thresholds`, returns per-feature deviation flags (`True`/`False`/`None` for undetermined) plus an `any_deviation` summary. Deep knee bend is tracked as an informational flag, not folded into `any_deviation` (a squat is not a posture fault). |

## Key decisions and their limitations (important — read before trusting outputs)

- **All angles are 2D, image-plane projections.** MediaPipe's z (depth) is not used:
  from a single RGB camera it is materially noisier than x/y, so including it would add
  noise rather than accuracy. This means every angle here is a camera-angle-dependent
  projection, not a true 3D biomechanical angle.
- **Forward-head angle assumes a side (sagittal) view.** It approximates the clinical
  craniovertebral angle (CVA) — angle between horizontal and the shoulder→ear line — a
  real physiotherapy heuristic where CVA < ~50° commonly flags forward head posture
  (hence the config default `forward_head_angle_deg: 50`). From a front-facing camera the
  shoulder-to-ear horizontal offset this relies on nearly vanishes, so this feature is
  only trustworthy with a side-view recording. The module picks whichever side (left/right
  ear+shoulder) is visible with higher confidence, so it degrades gracefully but cannot
  fix a front-facing camera's lack of depth information.
- **Shoulder/hip tilt and trunk/body inclination are most meaningful front-facing** (they
  measure left-right asymmetry and lean), and the reverse limitation applies: from a pure
  side view, left and right landmarks overlap and these numbers become noisy.
- **No single video angle supports every feature well.** This is exactly why thresholds
  live in `configs/config.yaml` rather than as code constants — a real deployment should
  calibrate thresholds (or simply ignore features unsupported by its camera placement)
  rather than trust these as universal cutoffs. This limitation is restated in the final
  technical report (Phase 9).
- **`None` means "undetermined," not "normal."** When required landmarks are missing,
  deviation flags are `None`, not `False` — `assess_posture` never silently treats missing
  data as "no deviation found."

## Testing performed

```bash
.venv/Scripts/python.exe -m pytest -q
```
36 tests pass (21 new this phase): geometry primitives verified against known
right-angle/straight-line/45°-lean configurations; knee-angle law-of-cosines verified
against a constructed right-angle bend and full extension; posture features verified for
level vs. tilted shoulders, upright vs. leaning trunk, missing-landmark passthrough, and
side-selection for the forward-head angle; posture engine verified for threshold
crossing in both directions, the deep-knee-bend/any_deviation separation, and
missing-data propagating as `None` rather than `False`.

These are geometry unit tests against synthetic coordinates — they confirm the math is
correct, not that the thresholds are clinically meaningful on real footage. That still
needs a real video (see Phase 2's open item).

## Git commands used this phase

```bash
git add -A
git commit -m "Phase 3: joint-angle calculation and rule-based posture feature extraction"
```

## Next (Phase 4, pending approval)

Dataset preparation and the first genuine ML training pipeline: UI-PRMD ingestion,
feature engineering for XGBoost, and the LSTM/GRU sequence model, with a real train/val/test
comparison.
