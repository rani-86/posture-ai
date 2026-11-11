"""Temporal smoothing and missing-landmark handling across a sequence of PoseFrames.

Two distinct problems are handled here:
1. Jitter: even a correctly detected landmark wobbles frame to frame. A moving average
   over a short window reduces this without materially lagging real movement, given the
   `smoothing_window` is small relative to typical posture-change timescales.
2. Missing landmarks: a landmark can be filtered out (low confidence) or absent entirely
   (no detection that frame). We hold the last known good value forward rather than
   interpolating or zero-filling, since zero-filling would silently inject a fake
   coordinate (0,0) into angle calculations, and forward-holding is this is the simplest,
   least assumption-laden choice before gap length is examined.
"""
from __future__ import annotations

from collections import deque

from src.pose_estimation.landmark_schema import Landmark, LandmarkName, PoseFrame


class LandmarkSmoother:
    """Stateful: feed `PoseFrame`s in temporal order, one at a time."""

    def __init__(self, window_size: int = 5):
        if window_size < 1:
            raise ValueError("window_size must be >= 1")
        self.window_size = window_size
        self._windows: dict[LandmarkName, deque[tuple[float, float, float]]] = {
            name: deque(maxlen=window_size) for name in LandmarkName
        }
        self._last_known: dict[LandmarkName, Landmark] = {}

    def smooth(self, pose_frame: PoseFrame) -> PoseFrame:
        smoothed: dict[LandmarkName, Landmark] = {}
        for name in LandmarkName:
            lm = pose_frame.landmarks[name]

            if lm.is_present:
                self._windows[name].append((lm.x, lm.y, lm.z))
                xs, ys, zs = zip(*self._windows[name])
                avg = Landmark(
                    x=sum(xs) / len(xs),
                    y=sum(ys) / len(ys),
                    z=sum(zs) / len(zs),
                    visibility=lm.visibility,
                )
                self._last_known[name] = avg
                smoothed[name] = avg
            elif name in self._last_known:
                # Hold last known value; visibility 0 so downstream code can tell this
                # frame's value for this landmark was carried forward, not observed.
                held = self._last_known[name]
                smoothed[name] = Landmark(held.x, held.y, held.z, visibility=0.0)
            else:
                smoothed[name] = Landmark(None, None, None, 0.0)

        return PoseFrame(
            frame_index=pose_frame.frame_index,
            timestamp_sec=pose_frame.timestamp_sec,
            detected=pose_frame.detected,
            landmarks=smoothed,
        )
