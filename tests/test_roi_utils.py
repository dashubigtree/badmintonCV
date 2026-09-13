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


if __name__ == "__main__":
    unittest.main()
