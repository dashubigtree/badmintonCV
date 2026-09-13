from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.roi_utils import (
    ROIConfigError,
    bottom_center,
    load_court_polygon,
    point_in_polygon,
    validate_polygon,
)
from scripts.annotate_court_roi import write_roi_config
from scripts.court_minimap import (
    COURT_LENGTH_M,
    COURT_WIDTH_M,
    THREE_POINT_FAR_LABELS,
    apply_homography,
    load_court_calibration,
    minimap_pixel,
    point_is_inside_court,
    three_point_far_court_points,
    write_court_calibration,
    write_three_point_far_calibration,
)
from scripts.pose_utils import visible_pose_points
from scripts.track_players import parse_args


class ROIUtilsTest(unittest.TestCase):
    def test_bottom_center_uses_lower_middle_of_box(self) -> None:
        self.assertEqual(bottom_center((10, 20, 30, 80)), (20, 80))

    def test_point_in_polygon_accepts_inside_and_boundary_points(self) -> None:
        polygon = [(0, 0), (10, 0), (10, 10), (0, 10)]
        self.assertTrue(point_in_polygon((5, 5), polygon))
        self.assertTrue(point_in_polygon((10, 5), polygon))
        self.assertFalse(point_in_polygon((11, 5), polygon))

    def test_validate_polygon_rejects_too_few_points(self) -> None:
        with self.assertRaises(ROIConfigError):
            validate_polygon([[0, 0], [1, 1]])

    def test_load_court_polygon_reads_json_config(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "court_roi.json"
            path.write_text(
                json.dumps({"court_polygon": [[0, 0], [10, 0], [0, 10]]}),
                encoding="utf-8",
            )
            self.assertEqual(
                load_court_polygon(path),
                [(0.0, 0.0), (10.0, 0.0), (0.0, 10.0)],
            )

    def test_write_roi_config_creates_expected_json(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "court_roi.json"
            write_roi_config([(1, 2), (3, 4), (5, 6)], path)
            data = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(data["court_polygon"], [[1, 2], [3, 4], [5, 6]])
            self.assertEqual(data["anchor"], "bottom_center")

    def test_visible_pose_points_keeps_requested_joints_above_confidence(self) -> None:
        xy = [[0, 0] for _ in range(17)]
        xy[0] = [10.2, 20.7]
        xy[5] = [30, 40]
        xy[6] = [50, 60]
        conf = [0.0 for _ in range(17)]
        conf[0] = 0.95
        conf[5] = 0.80
        conf[6] = 0.10

        points = visible_pose_points(xy, conf, min_confidence=0.30)

        self.assertEqual(points[0], (10, 21, "Head"))
        self.assertEqual(points[5], (30, 40, "L Sho"))
        self.assertNotIn(6, points)

    def test_track_players_parse_args_accepts_minimal_overlay(self) -> None:
        import sys
        from unittest.mock import patch

        test_args = [
            "track_players.py",
            "--video",
            "data/input/sample.mp4",
            "--roi",
            "configs/court_roi.json",
            "--output",
            "data/output/out.mp4",
            "--minimal-overlay",
        ]

        with patch.object(sys, "argv", test_args):
            args = parse_args()

        self.assertTrue(args.minimal_overlay)

    def test_write_and_load_court_calibration(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "court_calibration.json"
            write_court_calibration([(10, 20), (110, 20), (110, 220), (10, 220)], path)

            calibration = load_court_calibration(path)

            self.assertEqual(calibration["image_points"][0], (10.0, 20.0))
            self.assertEqual(calibration["court_points_m"][2], (COURT_WIDTH_M, COURT_LENGTH_M))

    def test_write_and_load_three_point_far_calibration(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "court_calibration.json"
            write_three_point_far_calibration([(100, 120), (40, 300), (460, 300)], path)
            data = json.loads(path.read_text(encoding="utf-8"))

            self.assertEqual(data["point_order"], list(THREE_POINT_FAR_LABELS))
            self.assertEqual(data["transform_type"], "affine")

            calibration = load_court_calibration(path)
            self.assertEqual(calibration["image_points"][1], (40.0, 300.0))
            self.assertEqual(calibration["court_points_m"], three_point_far_court_points())

    def test_apply_homography_identity(self) -> None:
        homography = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        self.assertEqual(apply_homography((3.5, 4.5), homography), (3.5, 4.5))

    def test_minimap_pixel_maps_court_center(self) -> None:
        pixel = minimap_pixel(
            (COURT_WIDTH_M / 2.0, COURT_LENGTH_M / 2.0),
            origin=(100, 20),
            size=(122, 268),
            court_width_m=COURT_WIDTH_M,
            court_length_m=COURT_LENGTH_M,
            padding=10,
        )
        self.assertEqual(pixel, (161, 154))
        self.assertTrue(point_is_inside_court((COURT_WIDTH_M, COURT_LENGTH_M), COURT_WIDTH_M, COURT_LENGTH_M))
        self.assertFalse(point_is_inside_court((COURT_WIDTH_M + 0.1, 0.0), COURT_WIDTH_M, COURT_LENGTH_M))

    def test_track_players_parse_args_accepts_minimap_options(self) -> None:
        import sys
        from unittest.mock import patch

        test_args = [
            "track_players.py",
            "--video",
            "data/input/sample.mp4",
            "--roi",
            "configs/court_roi.json",
            "--output",
            "data/output/out.mp4",
            "--minimap",
            "--court-calibration",
            "configs/court_calibration.json",
            "--trail-length",
            "30",
        ]

        with patch.object(sys, "argv", test_args):
            args = parse_args()

        self.assertTrue(args.minimap)
        self.assertEqual(args.court_calibration, "configs/court_calibration.json")
        self.assertEqual(args.trail_length, 30)


if __name__ == "__main__":
    unittest.main()
