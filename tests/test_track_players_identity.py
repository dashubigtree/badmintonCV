from __future__ import annotations

import unittest

from scripts.pose_utils import TARGET_POSE_POINTS
from scripts.track_players_identity import CSV_FIELDS, pose_csv_values


class PoseCsvValuesTest(unittest.TestCase):
    def test_exports_every_target_point_with_its_raw_confidence(self) -> None:
        pose_xy = [[float(index * 10), float(index * 10 + 1)] for index in range(17)]
        pose_conf = [index / 20.0 for index in range(17)]

        values = pose_csv_values(pose_xy, pose_conf)

        self.assertEqual(len(values), len(TARGET_POSE_POINTS) * 3)
        self.assertEqual(values[:3], [0.0, 1.0, 0.0])
        self.assertEqual(values[-3:], [160.0, 161.0, 0.8])
        self.assertEqual(len(CSV_FIELDS), 20 + len(values))

    def test_blanks_points_when_pose_is_unavailable(self) -> None:
        self.assertEqual(pose_csv_values(None, None), ["", "", ""] * len(TARGET_POSE_POINTS))


if __name__ == "__main__":
    unittest.main()
