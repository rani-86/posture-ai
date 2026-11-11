"""Coordinate normalization: make landmarks invariant to camera distance and frame size.

MediaPipe's x/y are already normalized to [0,1] by image width/height, but that alone
does not make two videos comparable: a person standing closer to the camera, or a video
shot in portrait vs landscape, still produces different absolute landmark positions for
the same posture. We re-center on the mid-hip point and rescale by torso length (shoulder
midpoint to hip midpoint distance) so downstream joint-angle and posture features are
roughly comparable across subjects, distances, and frame sizes. This does not correct for
camera viewing angle (e.g. side-view vs 3/4-view) - that limitation is documented in the
technical report, not fixed here.
"""
from __future__ import annotations

import math

from src.pose_estimation.landmark_schema import Landmark, LandmarkName, PoseFrame


def normalize_pose_frame(pose_frame: PoseFrame) -> PoseFrame:
    lm = pose_frame.landmarks
    left_hip, right_hip = lm[LandmarkName.LEFT_HIP], lm[LandmarkName.RIGHT_HIP]
    left_shoulder, right_shoulder = lm[LandmarkName.LEFT_SHOULDER], lm[LandmarkName.RIGHT_SHOULDER]

    required = (left_hip, right_hip, left_shoulder, right_shoulder)
    if not all(p.is_present for p in required):
        # Can't establish a reference frame this frame; pass landmarks through unchanged.
        return pose_frame

    mid_hip_x = (left_hip.x + right_hip.x) / 2
    mid_hip_y = (left_hip.y + right_hip.y) / 2
    mid_shoulder_x = (left_shoulder.x + right_shoulder.x) / 2
    mid_shoulder_y = (left_shoulder.y + right_shoulder.y) / 2

    torso_length = math.hypot(mid_shoulder_x - mid_hip_x, mid_shoulder_y - mid_hip_y)
    if torso_length < 1e-6:
        return pose_frame

    normalized: dict[LandmarkName, Landmark] = {}
    for name, point in lm.items():
        if not point.is_present:
            normalized[name] = point
            continue
        normalized[name] = Landmark(
            x=(point.x - mid_hip_x) / torso_length,
            y=(point.y - mid_hip_y) / torso_length,
            z=point.z / torso_length if point.z is not None else None,
            visibility=point.visibility,
        )

    return PoseFrame(
        frame_index=pose_frame.frame_index,
        timestamp_sec=pose_frame.timestamp_sec,
        detected=pose_frame.detected,
        landmarks=normalized,
    )
