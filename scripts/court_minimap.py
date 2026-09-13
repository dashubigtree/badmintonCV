from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable


COURT_WIDTH_M = 6.10
COURT_LENGTH_M = 13.40
COURT_CORNER_NAMES = ("top_left", "top_right", "bottom_right", "bottom_left")
Point = tuple[float, float]


class CourtCalibrationError(ValueError):
    """Raised when court calibration data cannot be used."""


def standard_court_points(width_m: float = COURT_WIDTH_M, length_m: float = COURT_LENGTH_M) -> list[Point]:
    return [(0.0, 0.0), (width_m, 0.0), (width_m, length_m), (0.0, length_m)]


def write_court_calibration(
    image_points: Iterable[tuple[int, int]],
    output_path: str | Path,
    width_m: float = COURT_WIDTH_M,
    length_m: float = COURT_LENGTH_M,
) -> None:
    points = list(image_points)
    if len(points) != 4:
        raise CourtCalibrationError("Court calibration requires exactly 4 corner points.")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "court_size_m": {"width": width_m, "length": length_m},
        "point_order": list(COURT_CORNER_NAMES),
        "image_points": [[int(x), int(y)] for x, y in points],
        "court_points_m": [[x, y] for x, y in standard_court_points(width_m, length_m)],
    }
    output.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_court_calibration(config_path: str | Path) -> dict[str, object]:
    path = Path(config_path)
    if not path.exists():
        raise CourtCalibrationError(f"Court calibration config not found: {path}")
    if path.stat().st_size == 0:
        raise CourtCalibrationError(f"Court calibration config is empty: {path}")

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise CourtCalibrationError(f"Court calibration config is not valid JSON: {path}\n{exc}") from exc

    image_points = validate_points(data.get("image_points"), "image_points")
    court_points = validate_points(data.get("court_points_m"), "court_points_m")
    if len(image_points) != 4 or len(court_points) != 4:
        raise CourtCalibrationError("image_points and court_points_m must each contain exactly 4 points.")

    court_size = data.get("court_size_m")
    if not isinstance(court_size, dict):
        raise CourtCalibrationError("court_size_m must be an object with width and length.")
    width = court_size.get("width")
    length = court_size.get("length")
    if not isinstance(width, (int, float)) or not isinstance(length, (int, float)):
        raise CourtCalibrationError("court_size_m.width and court_size_m.length must be numeric.")

    return {
        "court_size_m": {"width": float(width), "length": float(length)},
        "image_points": image_points,
        "court_points_m": court_points,
    }


def validate_points(points: object, field_name: str) -> list[Point]:
    if not isinstance(points, list):
        raise CourtCalibrationError(f"{field_name} must be a list of points.")

    validated: list[Point] = []
    for index, point in enumerate(points):
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            raise CourtCalibrationError(f"{field_name}[{index}] must be [x, y].")
        x, y = point
        if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
            raise CourtCalibrationError(f"{field_name}[{index}] must contain numeric values.")
        validated.append((float(x), float(y)))
    return validated


def apply_homography(point: Point, homography: object) -> Point:
    h = homography
    x, y = point
    denominator = h[2][0] * x + h[2][1] * y + h[2][2]
    if abs(float(denominator)) < 1e-9:
        raise CourtCalibrationError("Homography projection denominator is too close to zero.")
    mapped_x = (h[0][0] * x + h[0][1] * y + h[0][2]) / denominator
    mapped_y = (h[1][0] * x + h[1][1] * y + h[1][2]) / denominator
    return (float(mapped_x), float(mapped_y))


def point_is_inside_court(point_m: Point, width_m: float, length_m: float) -> bool:
    x, y = point_m
    return 0.0 <= x <= width_m and 0.0 <= y <= length_m


def minimap_pixel(
    point_m: Point,
    origin: tuple[int, int],
    size: tuple[int, int],
    court_width_m: float,
    court_length_m: float,
    padding: int,
) -> tuple[int, int]:
    x, y = point_m
    origin_x, origin_y = origin
    width_px, height_px = size
    drawable_w = max(1, width_px - padding * 2)
    drawable_h = max(1, height_px - padding * 2)
    px = origin_x + padding + int(round((x / court_width_m) * drawable_w))
    py = origin_y + padding + int(round((y / court_length_m) * drawable_h))
    return px, py


def stable_track_color(track_id: int) -> tuple[int, int, int]:
    palette = (
        (80, 220, 255),
        (255, 170, 80),
        (120, 255, 120),
        (255, 120, 210),
        (210, 210, 255),
        (80, 140, 255),
    )
    return palette[track_id % len(palette)]
