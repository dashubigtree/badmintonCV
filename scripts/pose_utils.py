from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class PosePoint:
    index: int
    name: str
    label: str


TARGET_POSE_POINTS: tuple[PosePoint, ...] = (
    PosePoint(0, "head", "Head"),
    PosePoint(5, "left_shoulder", "L Sho"),
    PosePoint(6, "right_shoulder", "R Sho"),
    PosePoint(7, "left_elbow", "L Elb"),
    PosePoint(8, "right_elbow", "R Elb"),
    PosePoint(9, "left_wrist", "L Wri"),
    PosePoint(10, "right_wrist", "R Wri"),
    PosePoint(11, "left_hip", "L Hip"),
    PosePoint(12, "right_hip", "R Hip"),
    PosePoint(13, "left_knee", "L Knee"),
    PosePoint(14, "right_knee", "R Knee"),
    PosePoint(15, "left_ankle", "L Ank"),
    PosePoint(16, "right_ankle", "R Ank"),
)

POSE_SKELETON_EDGES: tuple[tuple[int, int], ...] = (
    (5, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
)


def visible_pose_points(
    xy: Sequence[Sequence[float]],
    conf: Sequence[float] | None = None,
    min_confidence: float = 0.30,
) -> dict[int, tuple[int, int, str]]:
    points: dict[int, tuple[int, int, str]] = {}

    for pose_point in TARGET_POSE_POINTS:
        if pose_point.index >= len(xy):
            continue
        if conf is not None and pose_point.index < len(conf):
            if float(conf[pose_point.index]) < min_confidence:
                continue

        x, y = xy[pose_point.index]
        if float(x) <= 0 and float(y) <= 0:
            continue
        points[pose_point.index] = (int(round(float(x))), int(round(float(y))), pose_point.label)

    return points
