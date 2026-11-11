"""Synthetic, procedurally-generated activity dataset for Phase 5 (sitting / standing /
walking / bending / squatting).

THIS IS SYNTHETIC PLACEHOLDER DATA, NOT REAL HUMAN MOVEMENT. No labeled video clips were
available when this phase was built (flagged as an open item since Phase 1 - registering
for a labeled activity dataset, or recording/sourcing real clips, is still needed). Rather
than skip the ML component or leave the training pipeline untested, this module generates
procedurally-plausible pose-landmark sequences per activity class via simple forward
kinematics (thigh/shin/trunk angles -> 2D landmark positions, in the same 15-landmark,
mid-hip-centered, torso-length-normalized frame produced by src/pose_estimation for real
video), with per-class characteristic motion and per-sample randomization.

What this validates: the data loading, feature engineering, training loops, and
evaluation code all work correctly end-to-end, in the SAME feature space the real
MediaPipe pipeline (Phase 2/3) produces - so swapping in real clips later is a matter of
replacing this generator with a real loader, not rewriting the model code.
What this does NOT validate: real-world classification accuracy. Metrics reported from
this data describe how well these specific procedural templates can be told apart, not
how an activity-recognition model would perform on an actual person. This must be
restated wherever these numbers are reported (see docs/phase5_notes.md).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from src.pose_estimation.landmark_schema import LandmarkName

ACTIVITY_CLASSES = ["sitting", "standing", "walking", "bending", "squatting"]
NUM_ACTIVITY_CLASSES = len(ACTIVITY_CLASSES)
NUM_CHANNELS = len(LandmarkName) * 2  # x, y per landmark = 30

# Body segment lengths in torso-length units (mid_hip->mid_shoulder normalized to 1.0,
# consistent with src/pose_estimation/normalization.py). Not anatomically calibrated -
# chosen to produce plausible-looking proportions for a synthetic template.
_SHOULDER_HALF_WIDTH = 0.17
_HIP_HALF_WIDTH = 0.13
_UPPER_LEG = 0.55
_LOWER_LEG = 0.55
_UPPER_ARM = 0.35
_FOREARM = 0.3
_HEAD_OFFSET = 0.3
_EAR_HALF_WIDTH = 0.08


def _down(angle_deg: float) -> np.ndarray:
    """Unit vector pointing 'down' (increasing y) at `angle_deg` from vertical, positive
    = tilted toward +x."""
    rad = math.radians(angle_deg)
    return np.array([math.sin(rad), math.cos(rad)])


@dataclass
class ActivitySample:
    activity_id: int  # 0-indexed, see ACTIVITY_CLASSES
    subject_id: int  # synthetic "subject" group, for a leakage-free subject-wise split
    sequence: np.ndarray  # (T, 30), float32


def _build_frame(trunk_deg: float, hip_flex_deg: float, knee_flex_deg: float, arm_swing_deg: float) -> np.ndarray:
    """Forward-kinematics: joint angles -> (x, y) for all 15 landmarks, in
    mid-hip-centered, torso-length-normalized coordinates (matches Phase 2/3's frame)."""
    mid_hip = np.array([0.0, 0.0])
    trunk_dir = -_down(trunk_deg)  # trunk points "up" (toward -y) when trunk_deg small
    mid_shoulder = mid_hip + trunk_dir  # torso length = 1.0 by normalization convention

    left_shoulder = mid_shoulder + [-_SHOULDER_HALF_WIDTH, 0]
    right_shoulder = mid_shoulder + [_SHOULDER_HALF_WIDTH, 0]
    left_hip = mid_hip + [-_HIP_HALF_WIDTH, 0]
    right_hip = mid_hip + [_HIP_HALF_WIDTH, 0]

    nose = mid_shoulder + trunk_dir * _HEAD_OFFSET
    left_ear = nose + [-_EAR_HALF_WIDTH, 0.05]
    right_ear = nose + [_EAR_HALF_WIDTH, 0.05]

    # Legs: thigh direction from hip_flex (forward lean of thigh), shin folds back from
    # the thigh direction by (180 - knee_flex_deg) - knee_flex=180 means fully extended
    # (shin continues straight down from thigh), smaller means more bend.
    thigh_dir = _down(hip_flex_deg)
    shin_angle = hip_flex_deg - (180 - knee_flex_deg)
    shin_dir = _down(shin_angle)
    left_knee = left_hip + thigh_dir * _UPPER_LEG
    left_ankle = left_knee + shin_dir * _LOWER_LEG
    right_knee = right_hip + thigh_dir * _UPPER_LEG
    right_ankle = right_knee + shin_dir * _LOWER_LEG

    # Arms: simple pendulum swing around hanging-down, contralateral to legs for walking.
    left_arm_dir = _down(arm_swing_deg)
    right_arm_dir = _down(-arm_swing_deg)
    left_elbow = left_shoulder + left_arm_dir * _UPPER_ARM
    left_wrist = left_elbow + left_arm_dir * _FOREARM
    right_elbow = right_shoulder + right_arm_dir * _UPPER_ARM
    right_wrist = right_elbow + right_arm_dir * _FOREARM

    points = {
        LandmarkName.NOSE: nose,
        LandmarkName.LEFT_EAR: left_ear,
        LandmarkName.RIGHT_EAR: right_ear,
        LandmarkName.LEFT_SHOULDER: left_shoulder,
        LandmarkName.RIGHT_SHOULDER: right_shoulder,
        LandmarkName.LEFT_ELBOW: left_elbow,
        LandmarkName.RIGHT_ELBOW: right_elbow,
        LandmarkName.LEFT_WRIST: left_wrist,
        LandmarkName.RIGHT_WRIST: right_wrist,
        LandmarkName.LEFT_HIP: left_hip,
        LandmarkName.RIGHT_HIP: right_hip,
        LandmarkName.LEFT_KNEE: left_knee,
        LandmarkName.RIGHT_KNEE: right_knee,
        LandmarkName.LEFT_ANKLE: left_ankle,
        LandmarkName.RIGHT_ANKLE: right_ankle,
    }
    return np.concatenate([points[name] for name in LandmarkName]).astype(np.float32)


def _generate_episode(activity_id: int, num_frames: int, rng: np.random.Generator) -> np.ndarray:
    name = ACTIVITY_CLASSES[activity_id]
    t = np.arange(num_frames)
    noise_scale = 0.01

    if name == "standing":
        trunk = rng.uniform(-3, 3) + rng.normal(0, 1, num_frames) * 0.5
        hip_flex = rng.uniform(-3, 3) + rng.normal(0, 1, num_frames) * 0.5
        knee_flex = rng.uniform(172, 180) + rng.normal(0, 1, num_frames) * 0.5
        arm_swing = rng.normal(0, 1, num_frames) * 0.5
    elif name == "sitting":
        trunk = rng.uniform(2, 15) + rng.normal(0, 1, num_frames) * 0.5
        hip_flex = rng.uniform(65, 90) + rng.normal(0, 1, num_frames) * 0.5
        knee_flex = rng.uniform(80, 100) + rng.normal(0, 1, num_frames) * 0.5
        arm_swing = rng.normal(0, 1, num_frames) * 0.5
    elif name == "walking":
        freq = rng.uniform(0.15, 0.3)
        phase = rng.uniform(0, 2 * math.pi)
        hip_flex = 15 * np.sin(2 * math.pi * freq * t + phase) + rng.normal(0, 1.5, num_frames)
        knee_flex = 155 - 25 * np.sin(2 * math.pi * freq * t + phase + 0.6) ** 2 + rng.normal(0, 1.5, num_frames)
        arm_swing = -20 * np.sin(2 * math.pi * freq * t + phase) + rng.normal(0, 1.5, num_frames)
        trunk = rng.uniform(-2, 4) + 1.5 * np.sin(2 * math.pi * freq * t + phase) + rng.normal(0, 0.5, num_frames)
    elif name == "bending":
        ramp = np.clip(t / max(num_frames * 0.4, 1), 0, 1)
        hold = np.clip((t - num_frames * 0.6) / max(num_frames * 0.4, 1), 0, 1)
        shape = ramp - hold  # ramps up to ~1, holds, then ramps back down
        peak_angle = rng.uniform(55, 85)
        trunk = shape * peak_angle + rng.normal(0, 1, num_frames)
        hip_flex = shape * (peak_angle * 0.5) + rng.normal(0, 1, num_frames)
        knee_flex = 180 - shape * rng.uniform(5, 20) + rng.normal(0, 1, num_frames)
        arm_swing = rng.normal(0, 1, num_frames) * 0.5
    elif name == "squatting":
        freq = rng.uniform(0.08, 0.15)
        phase = rng.uniform(0, 2 * math.pi)
        depth = rng.uniform(80, 110)  # how deep the squat goes (degrees of knee flexion from standing)
        cycle = (1 - np.cos(2 * math.pi * freq * t + phase)) / 2  # 0 (standing) .. 1 (deep)
        knee_flex = 178 - cycle * depth + rng.normal(0, 1.5, num_frames)
        hip_flex = cycle * (depth * 0.4) + rng.normal(0, 1.5, num_frames)
        trunk = 5 + cycle * rng.uniform(10, 20) + rng.normal(0, 1, num_frames)
        arm_swing = -cycle * rng.uniform(10, 30) + rng.normal(0, 1, num_frames)
    else:
        raise ValueError(f"Unknown activity: {name}")

    frames = np.stack(
        [_build_frame(trunk[i], hip_flex[i], knee_flex[i], arm_swing[i]) for i in range(num_frames)]
    )
    frames += rng.normal(0, noise_scale, frames.shape).astype(np.float32)  # per-landmark jitter
    return frames.astype(np.float32)


def generate_synthetic_activity_dataset(
    num_subjects: int = 20,
    episodes_per_subject_per_class: int = 5,
    min_frames: int = 30,
    max_frames: int = 90,
    seed: int = 42,
) -> list[ActivitySample]:
    rng = np.random.default_rng(seed)
    samples = []
    for subject_id in range(1, num_subjects + 1):
        for activity_id in range(NUM_ACTIVITY_CLASSES):
            for _ in range(episodes_per_subject_per_class):
                num_frames = int(rng.integers(min_frames, max_frames + 1))
                sequence = _generate_episode(activity_id, num_frames, rng)
                samples.append(ActivitySample(activity_id=activity_id, subject_id=subject_id, sequence=sequence))
    return samples


def activity_label_fn(sample: ActivitySample) -> int:
    return sample.activity_id


def activity_subject_wise_split(
    samples: list[ActivitySample], train_subjects: set[int], val_subjects: set[int], test_subjects: set[int]
) -> tuple[list[ActivitySample], list[ActivitySample], list[ActivitySample]]:
    overlap = (train_subjects & val_subjects) | (train_subjects & test_subjects) | (val_subjects & test_subjects)
    if overlap:
        raise ValueError(f"Subject sets must be disjoint, overlap: {overlap}")
    train = [s for s in samples if s.subject_id in train_subjects]
    val = [s for s in samples if s.subject_id in val_subjects]
    test = [s for s in samples if s.subject_id in test_subjects]
    return train, val, test
