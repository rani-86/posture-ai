"""Rule-based posture deviation engine: compares PostureFeatures against configurable
thresholds (configs/config.yaml -> posture_thresholds) and flags likely deviations.

This is deliberately NOT a universal clinical diagnosis. A single fixed angle threshold
does not generalize across body types, camera angles, or contexts - the thresholds here
are starting points from common physiotherapy heuristics (e.g. CVA < 50 degrees as a
forward-head-posture indicator), kept configurable precisely so they can be recalibrated
per deployment rather than treated as ground truth. Every flag below is a label of
"this deviates from the configured threshold", not "this person has a postural disorder".
"""
from __future__ import annotations

from dataclasses import dataclass

from src.posture_analysis.posture_features import PostureFeatures
from src.utils.config import PostureThresholds


@dataclass
class PostureDeviationResult:
    features: PostureFeatures
    forward_head_deviation: bool | None  # None = undetermined (landmarks missing)
    shoulder_asymmetry_deviation: bool | None
    trunk_inclination_deviation: bool | None
    deep_knee_bend: bool | None  # informational flag, not inherently "bad"
    any_deviation: bool | None


def assess_posture(features: PostureFeatures, thresholds: PostureThresholds) -> PostureDeviationResult:
    forward_head_deviation = (
        None
        if features.forward_head_angle_deg is None
        else features.forward_head_angle_deg < thresholds.forward_head_angle_deg
    )

    shoulder_asymmetry_deviation = (
        None
        if features.shoulder_tilt_deg is None
        else features.shoulder_tilt_deg > thresholds.shoulder_asymmetry_deg
    )

    trunk_inclination_deviation = (
        None
        if features.trunk_inclination_deg is None
        else features.trunk_inclination_deg > thresholds.trunk_inclination_deg
    )

    knee_angles = [a for a in (features.left_knee_deg, features.right_knee_deg) if a is not None]
    deep_knee_bend = None if not knee_angles else min(knee_angles) < thresholds.knee_angle_deep_bend_deg

    determined_flags = [
        f
        for f in (forward_head_deviation, shoulder_asymmetry_deviation, trunk_inclination_deviation)
        if f is not None
    ]
    any_deviation = any(determined_flags) if determined_flags else None

    return PostureDeviationResult(
        features=features,
        forward_head_deviation=forward_head_deviation,
        shoulder_asymmetry_deviation=shoulder_asymmetry_deviation,
        trunk_inclination_deviation=trunk_inclination_deviation,
        deep_knee_bend=deep_knee_bend,
        any_deviation=any_deviation,
    )
