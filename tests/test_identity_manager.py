from __future__ import annotations

import unittest

from scripts.identity_manager import IdentityManager, PlayerState, get_ground_point


def make_detection(uid: int, raw_track_id: int, raw_x: float, raw_y: float, person_conf: float = 0.9) -> dict:
    return {
        "uid": uid,
        "raw_track_id": raw_track_id,
        "raw_x": raw_x,
        "raw_y": raw_y,
        "ground_method": "ankle_midpoint",
        "person_conf": person_conf,
    }


class GetGroundPointTest(unittest.TestCase):
    def test_prefers_ankle_midpoint_when_both_confident(self) -> None:
        keypoints = [[0.0, 0.0] for _ in range(17)]
        keypoints[15] = [100.0, 200.0]
        keypoints[16] = [110.0, 202.0]
        confidences = [0.0] * 17
        confidences[15] = 0.9
        confidences[16] = 0.9

        result = get_ground_point((50, 50, 150, 250), keypoints, confidences)

        self.assertEqual(result["method"], "ankle_midpoint")
        self.assertEqual((result["x"], result["y"]), (105.0, 201.0))

    def test_falls_back_to_single_ankle_when_other_is_low_confidence(self) -> None:
        keypoints = [[0.0, 0.0] for _ in range(17)]
        keypoints[15] = [100.0, 200.0]
        keypoints[16] = [110.0, 202.0]
        confidences = [0.0] * 17
        confidences[15] = 0.9
        confidences[16] = 0.05

        result = get_ground_point((50, 50, 150, 250), keypoints, confidences)

        self.assertEqual(result["method"], "left_ankle")
        self.assertEqual((result["x"], result["y"]), (100.0, 200.0))

    def test_falls_back_to_bbox_bottom_when_no_ankle_confident(self) -> None:
        keypoints = [[0.0, 0.0] for _ in range(17)]
        confidences = [0.0] * 17

        result = get_ground_point((50, 50, 150, 250), keypoints, confidences)

        self.assertEqual(result["method"], "bbox_bottom")
        self.assertEqual((result["x"], result["y"]), (100.0, 250.0))


class PlayerStateTest(unittest.TestCase):
    def test_rejects_single_frame_motion_spike_but_keeps_raw_measurement(self) -> None:
        state = PlayerState(1, "far")
        state.update(make_detection(0, 10, 0.0, 0.0), frame_index=0, max_speed_m_per_frame=0.30)

        info = state.update(make_detection(0, 10, 10.0, 10.0), frame_index=1, max_speed_m_per_frame=0.30)

        self.assertEqual(info["ground_status"], "corrected_motion_outlier")
        self.assertEqual((state.corrected_x, state.corrected_y), (0.0, 0.0))
        self.assertEqual((state.raw_x, state.raw_y), (10.0, 10.0))

    def test_accepts_motion_within_speed_bound_after_a_rejected_spike(self) -> None:
        state = PlayerState(1, "far")
        state.update(make_detection(0, 10, 0.0, 0.0), frame_index=0, max_speed_m_per_frame=0.30)
        state.update(make_detection(0, 10, 10.0, 10.0), frame_index=1, max_speed_m_per_frame=0.30)

        info = state.update(make_detection(0, 10, 0.2, 0.2), frame_index=2, max_speed_m_per_frame=0.30)

        self.assertEqual(info["ground_status"], "accepted")
        self.assertEqual((state.corrected_x, state.corrected_y), (0.2, 0.2))


class IdentityManagerTest(unittest.TestCase):
    def test_singles_only_allows_far_and_near_single_slots(self) -> None:
        manager = IdentityManager(match_format="singles", net_y_m=6.70)
        self.assertEqual(manager.allowed_player_ids, {1, 3})

    def test_doubles_allows_all_four_slots(self) -> None:
        manager = IdentityManager(match_format="doubles", net_y_m=6.70)
        self.assertEqual(manager.allowed_player_ids, {1, 2, 3, 4})

    def test_side_compatible_blocks_cross_net_match(self) -> None:
        manager = IdentityManager(match_format="doubles", net_y_m=6.70, side_tolerance_m=0.65)
        near_side_detection = make_detection(0, 10, 3.0, 12.0)

        self.assertFalse(manager.side_compatible("far", near_side_detection))
        self.assertTrue(manager.side_compatible("near", near_side_detection))

    def test_new_players_are_routed_to_the_correct_half_court(self) -> None:
        manager = IdentityManager(match_format="doubles", net_y_m=6.70)
        detections = [make_detection(0, 10, 3.0, 1.0), make_detection(1, 20, 3.0, 12.0)]

        assignments, _predicted = manager.assign(detections, frame_index=0)

        by_player = {item["player_id"]: item["detection"]["raw_track_id"] for item in assignments}
        self.assertEqual(by_player.get(1), 10)
        self.assertEqual(by_player.get(3), 20)

    def test_raw_id_ownership_blocks_a_second_state_from_stealing_it(self) -> None:
        manager = IdentityManager(match_format="doubles", net_y_m=6.70)
        manager.states[1].update(make_detection(0, 10, 3.0, 1.0), frame_index=0, max_speed_m_per_frame=0.30)

        stolen_detection = make_detection(0, 10, 3.0, 1.2)
        self.assertTrue(manager.ownership_compatible(manager.states[1], stolen_detection))
        self.assertFalse(manager.ownership_compatible(manager.states[2], stolen_detection))

    def test_relaxed_reacquire_reconnects_sole_unmatched_pair_beyond_strict_threshold(self) -> None:
        manager = IdentityManager(match_format="doubles", net_y_m=6.70)
        manager.assign([make_detection(0, 10, 3.0, 1.0)], frame_index=0)

        # ByteTrack drops the detection for a couple of frames, then a new raw id
        # appears too far from the predicted point for strict matching to accept
        # (missing=2 frames -> strict threshold is 1.00 + 0.07*2 = 1.14 m).
        manager.assign([], frame_index=1)
        assignments, _predicted = manager.assign([make_detection(0, 99, 5.0, 1.0)], frame_index=2)

        self.assertEqual(len(assignments), 1)
        self.assertEqual(assignments[0]["player_id"], 1)
        self.assertEqual(assignments[0]["match_mode"], "relaxed_reacquire")


if __name__ == "__main__":
    unittest.main()
