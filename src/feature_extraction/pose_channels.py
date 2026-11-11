"""Flattens a PoseFrame into the fixed 30-channel (x, y per landmark, in LandmarkName
enum order) layout that src/data/activity_dataset.py's synthetic generator also produces
- this is what makes the Phase 5 activity models directly usable on real pipeline output.

Missing landmarks are filled with 0.0 (the normalized-frame origin) rather than left out,
since the model needs a fixed-width input every frame. This is a coarser fallback than
Phase 2's temporal hold-forward (which already covers short gaps) - 0.0 only gets used
if a landmark has never been observed at all in the smoothing window, which should be
rare in a real video with a person in frame. Documented as a known rough edge, not a
silent approximation.
"""
from __future__ import annotations

import numpy as np

from src.pose_estimation.landmark_schema import LandmarkName, PoseFrame


def pose_frame_to_channels(pose_frame: PoseFrame) -> np.ndarray:
    values = []
    for name in LandmarkName:
        lm = pose_frame.landmarks[name]
        values.append(lm.x if lm.is_present else 0.0)
        values.append(lm.y if lm.is_present else 0.0)
    return np.array(values, dtype=np.float32)
