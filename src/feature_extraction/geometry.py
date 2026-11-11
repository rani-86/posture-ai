"""Pure 2D geometry helpers used by joint-angle and posture-feature calculations.

All angle math here operates on normalized image-plane (x, y) coordinates where y
increases downward (OpenCV/MediaPipe convention) - "vertical" and "horizontal" below are
defined relative to that convention, not real-world 3D space. MediaPipe's z (depth) is
deliberately not used for these angles: it is substantially noisier than x/y for a single
RGB camera (no stereo/depth sensor), so including it would add noise rather than
accuracy. This means all angles here are 2D projections and are sensitive to camera
viewing angle - documented further in posture_analysis.
"""
from __future__ import annotations

import math
from typing import NamedTuple

from src.pose_estimation.landmark_schema import Landmark


class Point2D(NamedTuple):
    x: float
    y: float


def to_point(landmark: Landmark) -> Point2D:
    if not landmark.is_present:
        raise ValueError("Cannot convert an absent landmark to a point")
    return Point2D(landmark.x, landmark.y)


def midpoint(a: Point2D, b: Point2D) -> Point2D:
    return Point2D((a.x + b.x) / 2, (a.y + b.y) / 2)


def vector(a: Point2D, b: Point2D) -> Point2D:
    """Vector from a to b."""
    return Point2D(b.x - a.x, b.y - a.y)


def vector_length(v: Point2D) -> float:
    return math.hypot(v.x, v.y)


def angle_at_vertex(a: Point2D, b: Point2D, c: Point2D) -> float:
    """Interior angle at vertex `b`, formed by points a-b-c, in degrees [0, 180].

    Computed via the law of cosines on the triangle a-b-c (equivalently, the angle
    between vectors b->a and b->c). Used for classic joint angles such as the knee
    (hip-knee-ankle) or elbow (shoulder-elbow-wrist).
    """
    ba = vector(b, a)
    bc = vector(b, c)
    len_ba, len_bc = vector_length(ba), vector_length(bc)
    if len_ba < 1e-9 or len_bc < 1e-9:
        raise ValueError("Degenerate triangle: coincident points")

    cos_angle = (ba.x * bc.x + ba.y * bc.y) / (len_ba * len_bc)
    cos_angle = max(-1.0, min(1.0, cos_angle))  # clamp for float error
    return math.degrees(math.acos(cos_angle))


def angle_from_vertical(v: Point2D) -> float:
    """Angle in degrees [0, 90] between vector `v` and the upward vertical axis (0, -1).

    0 degrees means `v` points straight up the image (e.g. a perfectly upright trunk);
    larger values mean more lean/tilt away from vertical, regardless of which side.
    """
    up = Point2D(0, -1)
    length = vector_length(v)
    if length < 1e-9:
        raise ValueError("Degenerate vector")
    cos_angle = (v.x * up.x + v.y * up.y) / length
    cos_angle = max(-1.0, min(1.0, cos_angle))
    angle = math.degrees(math.acos(cos_angle))
    return min(angle, 180 - angle)  # fold to [0, 90]: "leaning" is symmetric left/right


def angle_from_horizontal(v: Point2D) -> float:
    """Angle in degrees [0, 90] between vector `v` and the horizontal axis, direction-agnostic."""
    length = vector_length(v)
    if length < 1e-9:
        raise ValueError("Degenerate vector")
    return math.degrees(math.atan2(abs(v.y), abs(v.x)))
