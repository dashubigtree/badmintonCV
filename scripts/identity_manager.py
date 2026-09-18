"""Persistent player identity manager ported from badminton-tracker's
``track_players_identity.py``.

The existing ``MinimapIdentityStabilizer`` in this repo assigns identities with
plain nearest-neighbour slot matching, which has no notion of which half of the
court a player belongs to and no motion model. That is the root cause of the
flickering / duplicate-point behaviour noted in ``README.md``. This module
replaces that approach with three additional constraints:

- half-court side compatibility (far/near), so a P# can never suddenly match a
  detection across the net
- constant-velocity prediction, so short occlusions are bridged instead of
  causing a jump
- raw ByteTrack ID ownership protection, so a track ID already claimed by an
  active P# cannot be stolen by another P# on the same frame
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

Point = tuple[float, float]

ANKLE_LEFT_INDEX = 15
ANKLE_RIGHT_INDEX = 16

# Kept identical to badminton-tracker's defaults; not exposed as CLI flags
# there either, so left as constants here too.
EMA_ALPHA = 0.45
VELOCITY_ALPHA = 0.35
RAW_ID_BONUS_M = 0.65
SMOOTH_RESET_GAP_FRAMES = 12


def get_ground_point(
    box: Sequence[float],
    keypoints: Sequence[Sequence[float]],
    confidences: Sequence[float] | None,
    ankle_conf: float = 0.20,
) -> dict[str, object]:
    """Pick the player's ground contact point.

    Priority: ankle midpoint (both confident) > single confident ankle >
    bounding-box bottom-center fallback.
    """
    x1, _y1, x2, y2 = box
    left_ankle = keypoints[ANKLE_LEFT_INDEX]
    right_ankle = keypoints[ANKLE_RIGHT_INDEX]

    if confidences is not None:
        left_conf = float(confidences[ANKLE_LEFT_INDEX])
        right_conf = float(confidences[ANKLE_RIGHT_INDEX])
    else:
        left_conf = right_conf = 1.0

    left_valid = left_conf >= ankle_conf and float(left_ankle[0]) > 1 and float(left_ankle[1]) > 1
    right_valid = right_conf >= ankle_conf and float(right_ankle[0]) > 1 and float(right_ankle[1]) > 1

    if left_valid and right_valid:
        x = (float(left_ankle[0]) + float(right_ankle[0])) / 2.0
        y = (float(left_ankle[1]) + float(right_ankle[1])) / 2.0
        method = "ankle_midpoint"
    elif left_valid:
        x, y = float(left_ankle[0]), float(left_ankle[1])
        method = "left_ankle"
    elif right_valid:
        x, y = float(right_ankle[0]), float(right_ankle[1])
        method = "right_ankle"
    else:
        x, y = (float(x1) + float(x2)) / 2.0, float(y2)
        method = "bbox_bottom"

    return {"x": x, "y": y, "method": method, "left_conf": left_conf, "right_conf": right_conf}


def ground_quality_for_method(ground_method: str) -> str:
    if ground_method == "ankle_midpoint":
        return "two_ankles"
    if ground_method in ("left_ankle", "right_ankle"):
        return "single_ankle"
    return "bbox_fallback"


@dataclass
class PlayerState:
    """Tracks one logical player slot (P1..P4) across frames."""

    player_id: int
    side: str  # "far" or "near"
    active: bool = False
    last_seen_frame: int = -10**9
    last_raw_track_id: int | None = None
    raw_x: float | None = None
    raw_y: float | None = None
    # corrected_x/y is the last accepted motion origin; raw_x/y is always the
    # latest measurement even when rejected as a motion outlier.
    corrected_x: float | None = None
    corrected_y: float | None = None
    last_position_frame: int = -10**9
    stable_x: float | None = None
    stable_y: float | None = None
    vx: float = 0.0
    vy: float = 0.0

    def missing_frames(self, frame_index: int) -> int:
        if not self.active:
            return 10**9
        return max(0, frame_index - self.last_seen_frame)

    def predict(self, frame_index: int) -> Point | None:
        if not self.active or self.corrected_x is None or self.corrected_y is None:
            return None
        dt = max(0, frame_index - self.last_position_frame)
        return (self.corrected_x + self.vx * dt, self.corrected_y + self.vy * dt)

    def reset(self) -> None:
        self.active = False
        self.last_seen_frame = -10**9
        self.last_raw_track_id = None
        self.raw_x = self.raw_y = None
        self.corrected_x = self.corrected_y = None
        self.last_position_frame = -10**9
        self.stable_x = self.stable_y = None
        self.vx = self.vy = 0.0

    def update(self, detection: dict, frame_index: int, max_speed_m_per_frame: float) -> dict:
        """Ground Position Correction: reject single-frame motion spikes.

        Returns the fields needed for CSV logging / overlay text.
        """
        new_x = float(detection["raw_x"])
        new_y = float(detection["raw_y"])
        ground_quality = ground_quality_for_method(detection.get("ground_method", "unknown"))
        ground_status = "accepted"
        ground_speed = None

        if not self.active:
            self.active = True
            corrected_x, corrected_y = new_x, new_y
            self.stable_x, self.stable_y = new_x, new_y
            self.vx = self.vy = 0.0
            self.last_position_frame = frame_index
        else:
            dt_position = max(1, frame_index - self.last_position_frame)
            position_dx = new_x - self.corrected_x
            position_dy = new_y - self.corrected_y
            ground_speed = math.hypot(position_dx, position_dy) / dt_position

            reject_motion_outlier = (
                dt_position <= SMOOTH_RESET_GAP_FRAMES and ground_speed > max_speed_m_per_frame
            )

            if reject_motion_outlier:
                ground_status = "corrected_motion_outlier"
                predicted = self.predict(frame_index)
                corrected_x, corrected_y = predicted if predicted is not None else (self.corrected_x, self.corrected_y)
                self.stable_x = EMA_ALPHA * corrected_x + (1.0 - EMA_ALPHA) * self.stable_x
                self.stable_y = EMA_ALPHA * corrected_y + (1.0 - EMA_ALPHA) * self.stable_y
            else:
                observed_vx = position_dx / dt_position
                observed_vy = position_dy / dt_position
                speed = math.hypot(observed_vx, observed_vy)
                if speed > max_speed_m_per_frame:
                    scale = max_speed_m_per_frame / speed
                    observed_vx *= scale
                    observed_vy *= scale

                self.vx = VELOCITY_ALPHA * observed_vx + (1.0 - VELOCITY_ALPHA) * self.vx
                self.vy = VELOCITY_ALPHA * observed_vy + (1.0 - VELOCITY_ALPHA) * self.vy

                corrected_x, corrected_y = new_x, new_y
                if dt_position > SMOOTH_RESET_GAP_FRAMES:
                    self.stable_x, self.stable_y = new_x, new_y
                else:
                    self.stable_x = EMA_ALPHA * new_x + (1.0 - EMA_ALPHA) * self.stable_x
                    self.stable_y = EMA_ALPHA * new_y + (1.0 - EMA_ALPHA) * self.stable_y

                self.last_position_frame = frame_index

            self.corrected_x, self.corrected_y = corrected_x, corrected_y

        self.corrected_x, self.corrected_y = corrected_x, corrected_y
        self.raw_x, self.raw_y = new_x, new_y
        self.last_seen_frame = frame_index
        self.last_raw_track_id = detection["raw_track_id"]

        return {
            "ground_quality": ground_quality,
            "ground_status": ground_status,
            "corrected_x": corrected_x,
            "corrected_y": corrected_y,
            "ground_speed_m_per_frame": ground_speed,
        }


class IdentityManager:
    """Assigns detections to persistent P1..P4 identities frame by frame."""

    def __init__(
        self,
        match_format: str,
        net_y_m: float,
        side_tolerance_m: float = 0.65,
        max_match_distance_m: float = 3.20,
        max_speed_m_per_frame: float = 0.30,
        identity_timeout_frames: int = 150,
        predict_display_frames: int = 12,
    ) -> None:
        if match_format not in ("singles", "doubles"):
            raise ValueError(f"Unsupported match_format: {match_format!r}")

        self.match_format = match_format
        self.net_y_m = net_y_m
        self.side_tolerance_m = side_tolerance_m
        self.max_match_distance_m = max_match_distance_m
        self.max_speed_m_per_frame = max_speed_m_per_frame
        self.identity_timeout_frames = identity_timeout_frames
        self.predict_display_frames = predict_display_frames

        self.states: dict[int, PlayerState] = {
            1: PlayerState(1, "far"),
            2: PlayerState(2, "far"),
            3: PlayerState(3, "near"),
            4: PlayerState(4, "near"),
        }
        self.allowed_player_ids = {1, 3} if match_format == "singles" else {1, 2, 3, 4}

    def allowed_states(self) -> list[PlayerState]:
        return [state for player_id, state in self.states.items() if player_id in self.allowed_player_ids]

    def raw_id_owner(self, raw_track_id: int) -> PlayerState | None | bool:
        """Return the owning state, None if unowned, or False if ambiguously owned."""
        owners = [s for s in self.allowed_states() if s.active and s.last_raw_track_id == raw_track_id]
        if len(owners) == 1:
            return owners[0]
        if len(owners) > 1:
            return False
        return None

    def ownership_compatible(self, state: PlayerState, detection: dict) -> bool:
        owner = self.raw_id_owner(detection["raw_track_id"])
        if owner is None:
            return True
        if owner is False:
            return False
        return owner.player_id == state.player_id

    def detection_has_active_owner(self, detection: dict) -> bool:
        return self.raw_id_owner(detection["raw_track_id"]) is not None

    def side_compatible(self, side: str, detection: dict) -> bool:
        y = detection["raw_y"]
        if side == "far":
            return y <= self.net_y_m + self.side_tolerance_m
        return y >= self.net_y_m - self.side_tolerance_m

    def match_threshold(self, state: PlayerState, frame_index: int) -> float:
        missing = min(state.missing_frames(frame_index), 30)
        return min(self.max_match_distance_m, 1.00 + 0.07 * missing)

    def pair_cost(self, state: PlayerState, detection: dict, frame_index: int) -> float | None:
        if not self.ownership_compatible(state, detection):
            return None
        if not self.side_compatible(state.side, detection):
            return None

        predicted = state.predict(frame_index)
        if predicted is None:
            return None
        px, py = predicted
        distance = math.hypot(detection["raw_x"] - px, detection["raw_y"] - py)
        threshold = self.match_threshold(state, frame_index)

        if state.last_raw_track_id == detection["raw_track_id"]:
            threshold = max(threshold, 2.25)
            cost = max(0.0, distance - RAW_ID_BONUS_M)
        else:
            cost = distance

        if distance > threshold:
            return None
        return cost

    def best_matching(
        self,
        states: list[PlayerState],
        detections: list[dict],
        frame_index: int,
    ) -> list[tuple[PlayerState, int, float]]:
        """Brute-force search for the assignment with the most matches at
        lowest total cost. At most 4 states are ever passed in, so DFS is
        cheap and avoids pulling in a Hungarian-algorithm dependency."""
        best_pairs: list[tuple[PlayerState, int, float]] = []
        best_match_count = -1
        best_cost = float("inf")

        def search(state_index: int, used: set[int], pairs: list, total_cost: float) -> None:
            nonlocal best_pairs, best_match_count, best_cost

            if state_index >= len(states):
                count = len(pairs)
                if count > best_match_count or (count == best_match_count and total_cost < best_cost):
                    best_match_count = count
                    best_cost = total_cost
                    best_pairs = list(pairs)
                return

            state = states[state_index]
            search(state_index + 1, used, pairs, total_cost)

            for det_index, detection in enumerate(detections):
                if det_index in used:
                    continue
                cost = self.pair_cost(state, detection, frame_index)
                if cost is None:
                    continue
                used.add(det_index)
                pairs.append((state, det_index, cost))
                search(state_index + 1, used, pairs, total_cost + cost)
                pairs.pop()
                used.remove(det_index)

        search(0, set(), [], 0.0)
        return best_pairs

    def deactivate_stale(self, frame_index: int) -> None:
        for state in self.allowed_states():
            if state.active and state.missing_frames(frame_index) > self.identity_timeout_frames:
                state.reset()

    def assign(self, detections: list[dict], frame_index: int) -> tuple[list[dict], list[dict]]:
        self.deactivate_stale(frame_index)

        assignments: list[dict] = []
        used_detection_uids: set[int] = set()

        # 1. Strict matching for already-active states.
        for side in ("far", "near"):
            active_states = [s for s in self.allowed_states() if s.active and s.side == side]
            candidate_detections = [d for d in detections if self.side_compatible(side, d)]
            pairs = self.best_matching(active_states, candidate_detections, frame_index)

            for state, local_det_index, cost in pairs:
                detection = candidate_detections[local_det_index]
                if detection["uid"] in used_detection_uids:
                    continue
                position_info = state.update(detection, frame_index, self.max_speed_m_per_frame)
                used_detection_uids.add(detection["uid"])
                assignments.append(
                    {
                        "player_id": state.player_id,
                        "detection": detection,
                        "match_cost": cost,
                        "match_mode": "strict",
                        "position_info": position_info,
                    }
                )

        # 2. Relaxed reacquire: exactly one unmatched active state + one
        # unmatched detection on the same side -> reconnect directly. 2x2+
        # ambiguity is left alone to avoid swapping identities.
        for side in ("far", "near"):
            assigned_player_ids = {item["player_id"] for item in assignments}
            unmatched_states = [
                s for s in self.allowed_states()
                if s.active and s.side == side and s.player_id not in assigned_player_ids
            ]
            if len(unmatched_states) != 1:
                continue
            state = unmatched_states[0]

            unmatched_detections = [
                d for d in detections
                if d["uid"] not in used_detection_uids
                and self.side_compatible(side, d)
                and self.ownership_compatible(state, d)
            ]
            if len(unmatched_detections) != 1:
                continue
            detection = unmatched_detections[0]

            predicted_position = state.predict(frame_index)
            if predicted_position is None:
                continue
            match_cost = math.hypot(detection["raw_x"] - predicted_position[0], detection["raw_y"] - predicted_position[1])

            position_info = state.update(detection, frame_index, self.max_speed_m_per_frame)
            used_detection_uids.add(detection["uid"])
            assignments.append(
                {
                    "player_id": state.player_id,
                    "detection": detection,
                    "match_cost": match_cost,
                    "match_mode": "relaxed_reacquire",
                    "position_info": position_info,
                }
            )

        # 3. New logical players, only once a side has no active-but-unmatched
        # state left (avoids misbuilding a duplicate P# during a brief loss).
        unmatched = [d for d in detections if d["uid"] not in used_detection_uids]
        for side in ("far", "near"):
            assigned_player_ids = {item["player_id"] for item in assignments}
            has_unmatched_active_state = any(
                s.active and s.side == side and s.player_id not in assigned_player_ids
                for s in self.allowed_states()
            )
            if has_unmatched_active_state:
                continue

            side_detections = [
                d for d in unmatched
                if self.side_compatible(side, d) and not self.detection_has_active_owner(d)
            ]
            side_detections.sort(key=lambda d: d["person_conf"], reverse=True)

            inactive_states = sorted(
                (s for s in self.allowed_states() if not s.active and s.side == side),
                key=lambda s: s.player_id,
            )

            side_detections = side_detections[: len(inactive_states)]
            side_detections.sort(key=lambda d: d["raw_x"])

            for state, detection in zip(inactive_states, side_detections):
                position_info = state.update(detection, frame_index, self.max_speed_m_per_frame)
                used_detection_uids.add(detection["uid"])
                assignments.append(
                    {
                        "player_id": state.player_id,
                        "detection": detection,
                        "match_cost": 0.0,
                        "match_mode": "new_player",
                        "position_info": position_info,
                    }
                )

        # 4. Short occlusions: keep a motion-predicted position for players
        # not seen this frame, so the minimap doesn't blink.
        visible_ids = {item["player_id"] for item in assignments}
        predicted: list[dict] = []
        for state in self.allowed_states():
            if not state.active or state.player_id in visible_ids:
                continue
            missing = state.missing_frames(frame_index)
            if missing <= self.predict_display_frames:
                pred = state.predict(frame_index)
                if pred is not None:
                    predicted.append(
                        {"player_id": state.player_id, "pred_x": pred[0], "pred_y": pred[1], "missing_frames": missing}
                    )

        assignments.sort(key=lambda item: item["player_id"])
        predicted.sort(key=lambda item: item["player_id"])
        return assignments, predicted
