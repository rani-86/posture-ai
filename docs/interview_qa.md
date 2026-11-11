# PostureAI — Likely Interview Questions & Answers

Grounded in what was actually built in this project, not generic ML trivia — use these
to practice explaining real decisions you made, with real numbers you can cite.

## Computer Vision / Pose Estimation

**Q: Why MediaPipe Pose instead of OpenPose or a custom-trained pose model?**
A: MediaPipe Pose is optimized for real-time, single-person, CPU inference — the right
fit for a webcam-facing app where low latency matters more than multi-person tracking or
squeezing out marginal accuracy. Training a custom pose model from scratch would need a
large labeled pose dataset and GPU infrastructure disproportionate to this project's
actual requirement (track one person's joints).

**Q: You pinned `mediapipe==0.10.14` — why?**
A: An unpinned `pip install mediapipe` resolved to `1.2.2`, which dropped the legacy
`solutions.pose` API in favor of a different Tasks API. Pinning avoided a breaking API
mismatch mid-project, and keeps the dependency deterministic for anyone else running
`pip install -r requirements.txt`.

**Q: Why didn't you use MediaPipe's z (depth) coordinate in your angle calculations?**
A: A single RGB camera has no real depth sensor — MediaPipe's z is inferred, not
measured, and is materially noisier than x/y. Including it in angle math would add noise
rather than accuracy, so I deliberately kept all geometric features as 2D projections.
This is a stated limitation, not an oversight: it means every angle is sensitive to
camera viewing angle.

**Q: How do you handle a landmark MediaPipe fails to detect mid-video?**
A: Two layers. First, per-landmark confidence filtering: below a configurable visibility
threshold, a landmark is marked absent rather than trusted. Second, temporal smoothing
holds a missing landmark at its last known value for a short window (with visibility
forced to 0 so downstream code can distinguish "held" from "observed") rather than
zero-filling or interpolating — zero-filling would inject a fake (0,0) coordinate into
angle math, and interpolation needs look-ahead that breaks frame-by-frame streaming.

**Q: What's the "craniovertebral angle" and why does it matter here?**
A: It's a real physiotherapy measurement — the angle between a horizontal line through
the shoulder/C7 and the line to the ear — used to assess forward head posture (commonly
flagged below ~50°). I used it as the basis for my forward-head-angle feature, but it
only works from a side-view camera; from the front, the horizontal offset it depends on
nearly vanishes. I documented that limitation explicitly rather than presenting the
feature as universally reliable.

## Machine Learning / Deep Learning

**Q: Walk me through your model comparison methodology.**
A: XGBoost on fixed-length summary-statistic features (mean/std/min/max/range per
channel) vs. LSTM and GRU on the raw, padded sequences. All three evaluated with
identical metrics (accuracy, macro precision/recall/F1, confusion matrix) on the same
held-out subjects. On real UI-PRMD data: LSTM 0.940 accuracy/0.937 macro-F1, XGBoost
0.930/0.928, GRU 0.900/0.897.

**Q: Why split by subject instead of randomly?**
A: Repetitions from the same subject are correlated — their individual movement style
leaks across train/test in a random split, inflating test accuracy relative to
real-world performance on a genuinely new person. I split by subject (e.g. subjects 1-7
train, 8 val, 9-10 test) so no subject's data crosses a split boundary.

**Q: Your three models all confused the same two movement classes. How did you verify
that wasn't just a modeling failure?**
A: I computed a model-independent separability metric — normalized mean-feature
distance between the two confused classes (m07/m10) versus a typical never-confused
pair. It came out to 0.43 vs. 0.80: m07 and m10 are genuinely about half as separable in
this feature space. That's real, data-grounded evidence that the confusion reflects
exercise similarity, not a weakness specific to one architecture — more convincing than
just noting "all three models made the same mistake."

**Q: You reported 100% accuracy on your activity-recognition model. Should that worry
me?**
A: Yes, and I say so explicitly in my own docs. That model was trained and evaluated
only on synthetic, procedurally-generated data (no real labeled clips were available) —
the 5 classes are far more separable than real human movement would be. 100% on data
like that means the training/evaluation pipeline has no bugs, not that the model
recognizes activities well. I'd expect accuracy to drop substantially on real clips, and
I'd read that drop as the dataset becoming realistic, not as a regression.

**Q: LSTM vs. GRU vs. 1D CNN for this task — how would you choose?**
A: LSTM/GRU fit naturally here because sequence length varies a lot (39-160 frames in
UI-PRMD) and the task is "classify the whole sequence," which benefits from an explicit
running hidden state. A CNN assumes local structure — useful for images, and workable
for sequences with dilated convolutions, but needs more careful handling of variable
length. I used both LSTM and GRU as a genuine comparison (LSTM edged out GRU slightly
here) and didn't add a CNN baseline since it wasn't necessary to answer the actual
question I was investigating.

**Q: How did you prevent data leakage in your feature normalization?**
A: Per-channel z-score statistics (mean/std) are computed only on the training split and
explicitly passed into the validation/test dataset construction — never recomputed from
whatever data is currently being evaluated. I also found and fixed a real gap where
`train.py` saved model weights but not these normalization stats, which would have made
checkpoint-based inference (Phase 6/7) silently wrong; I added that persistence and
verified it by reloading checkpoints and reproducing identical metrics.

**Q: What would you do with more time/data?**
A: Get real labeled video clips for the 5 activity classes (the actual open blocker
flagged since Phase 1) and retrain the activity model on real data — comparing real vs.
synthetic performance would itself be an interesting result. For movement classification,
I'd pursue UI-PRMD's official joint-documentation to enable a real Kinect-vs-Vicon MPJPE
comparison, which I deliberately deferred rather than guess at.

## Software/ML Engineering

**Q: Why is your UI-PRMD movement classifier not exposed in your API, even though it
scores 93-94%?**
A: Feature-space mismatch. It was trained on UI-PRMD's 66-channel Kinect Euler-angle
output; my own MediaPipe pipeline produces a different, 30-channel feature space. Wiring
it into the API would mean feeding it the wrong kind of input in the right shape —
either a crash or, worse, a plausible-looking but meaningless prediction. I'd rather
leave a capability out, with the reasoning documented, than ship something that looks
like it works but doesn't.

**Q: Why does your Streamlit dashboard call your inference code directly instead of
going through your own FastAPI backend?**
A: Streamlit is already a single Python process; routing through its own sibling HTTP
API would add latency and a second server to keep running, for no benefit at this
project's scale. The API remains the integration point for a separate client; the
dashboard is a second, independent consumer of the same `analyze_video` function.

**Q: How do you keep training and inference separate?**
A: `train.py`/`train_activity.py` do the (expensive, deliberate) training and save
checkpoints plus normalization stats; `evaluate.py` loads only those checkpoints and
re-runs inference — no retraining. I used this to catch the missing-normalization-stats
bug: reloading a checkpoint and getting *different* numbers than the original training
run would have been the signal something was wrong; getting identical numbers confirmed
the save/load path was correct.
