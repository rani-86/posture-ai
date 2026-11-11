import math

import pytest

from src.feature_extraction.geometry import (
    Point2D,
    angle_at_vertex,
    angle_from_horizontal,
    angle_from_vertical,
    midpoint,
    vector,
)


def test_angle_at_vertex_right_angle():
    a, b, c = Point2D(0, 0), Point2D(0, 1), Point2D(1, 1)
    assert math.isclose(angle_at_vertex(a, b, c), 90.0, abs_tol=1e-6)


def test_angle_at_vertex_straight_line():
    a, b, c = Point2D(0, 0), Point2D(1, 0), Point2D(2, 0)
    assert math.isclose(angle_at_vertex(a, b, c), 180.0, abs_tol=1e-6)


def test_angle_at_vertex_degenerate_raises():
    with pytest.raises(ValueError):
        angle_at_vertex(Point2D(0, 0), Point2D(0, 0), Point2D(1, 1))


def test_angle_from_vertical_upright():
    v = vector(Point2D(0.5, 0.6), Point2D(0.5, 0.2))  # points straight up (y decreases)
    assert math.isclose(angle_from_vertical(v), 0.0, abs_tol=1e-6)


def test_angle_from_vertical_45_degrees():
    v = vector(Point2D(0, 1), Point2D(1, 0))  # up and to the right, equal magnitude
    assert math.isclose(angle_from_vertical(v), 45.0, abs_tol=1e-6)


def test_angle_from_horizontal_is_direction_agnostic():
    right = angle_from_horizontal(Point2D(1, 0.5))
    left = angle_from_horizontal(Point2D(-1, 0.5))
    assert math.isclose(right, left, abs_tol=1e-9)


def test_midpoint():
    assert midpoint(Point2D(0, 0), Point2D(2, 4)) == Point2D(1, 2)
