"""Shared helpers for badminton court ROI filtering."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable


Point = tuple[float, float]
Box = tuple[float, float, float, float]


class ROIConfigError(ValueError):
    """Raised when the ROI config cannot be used."""


def load_court_polygon(config_path: str | Path) -> list[Point]:
    path = Path(config_path)
    if not path.exists():
        raise ROIConfigError(f"ROI config not found: {path}")
    if path.stat().st_size == 0:
        raise ROIConfigError(
            f"ROI config is empty: {path}\n"
            'Expected JSON like: {"court_polygon": [[320, 180], [1580, 180], [1880, 1040], [80, 1040]]}'
        )

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ROIConfigError(f"ROI config is not valid JSON: {path}\n{exc}") from exc

    polygon = data.get("court_polygon")
    return validate_polygon(polygon)


def validate_polygon(polygon: object) -> list[Point]:
    if not isinstance(polygon, list) or len(polygon) < 3:
        raise ROIConfigError("court_polygon must contain at least 3 points.")

    points: list[Point] = []
    for index, point in enumerate(polygon):
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            raise ROIConfigError(f"Point {index} must be [x, y].")
        x, y = point
        if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
            raise ROIConfigError(f"Point {index} must contain numeric x and y values.")
        points.append((float(x), float(y)))
    return points


def bottom_center(box: Box) -> Point:
    x1, _y1, x2, y2 = box
    return ((x1 + x2) / 2.0, y2)


def point_in_polygon(point: Point, polygon: Iterable[Point]) -> bool:
    """Return True when point is inside or on the polygon boundary."""
    x, y = point
    points = list(polygon)
    inside = False

    j = len(points) - 1
    for i, (xi, yi) in enumerate(points):
        xj, yj = points[j]
        if _point_on_segment(point, (xi, yi), (xj, yj)):
            return True

        intersects = (yi > y) != (yj > y)
        if intersects:
            x_intersection = (xj - xi) * (y - yi) / (yj - yi) + xi
            if x <= x_intersection:
                inside = not inside
        j = i

    return inside


def _point_on_segment(point: Point, start: Point, end: Point, epsilon: float = 1e-9) -> bool:
    px, py = point
    x1, y1 = start
    x2, y2 = end

    cross = (py - y1) * (x2 - x1) - (px - x1) * (y2 - y1)
    if abs(cross) > epsilon:
        return False

    within_x = min(x1, x2) - epsilon <= px <= max(x1, x2) + epsilon
    within_y = min(y1, y2) - epsilon <= py <= max(y1, y2) + epsilon
    return within_x and within_y


def polygon_as_int_points(polygon: Iterable[Point]) -> list[tuple[int, int]]:
    return [(int(round(x)), int(round(y))) for x, y in polygon]
