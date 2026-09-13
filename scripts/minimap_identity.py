from __future__ import annotations

from dataclasses import dataclass, field


Point = tuple[float, float]


@dataclass
class MinimapIdentityStabilizer:
    max_players: int = 4
    stale_frames: int = 12
    match_distance_m: float = 1.8
    merge_distance_m: float = 0.45
    next_slot_id: int = 1
    raw_to_slot: dict[int, int] = field(default_factory=dict)
    slot_positions: dict[int, Point] = field(default_factory=dict)
    slot_last_seen: dict[int, int] = field(default_factory=dict)

    def update(self, raw_positions: dict[int, Point], frame_index: int) -> dict[int, Point]:
        self._remove_stale_slots(frame_index)
        display_positions: dict[int, Point] = {}
        used_slots: set[int] = set()
        merged_raw_positions = merge_close_raw_positions(raw_positions, self.merge_distance_m)

        for raw_id, point in sorted(merged_raw_positions.items()):
            slot_id = self._slot_for_raw_id(raw_id, point, frame_index, used_slots)
            if slot_id is None:
                continue

            self.raw_to_slot[raw_id] = slot_id
            self.slot_positions[slot_id] = point
            self.slot_last_seen[slot_id] = frame_index
            display_positions[slot_id] = point
            used_slots.add(slot_id)

        return display_positions

    def _slot_for_raw_id(
        self,
        raw_id: int,
        point: Point,
        frame_index: int,
        used_slots: set[int],
    ) -> int | None:
        existing_slot = self.raw_to_slot.get(raw_id)
        if existing_slot in self.slot_positions and existing_slot not in used_slots:
            return existing_slot

        reusable_slot = self._nearest_recent_slot(point, frame_index, used_slots)
        if reusable_slot is not None:
            return reusable_slot

        if len(self.slot_positions) >= self.max_players:
            return None

        return self._new_slot_id()

    def _nearest_recent_slot(self, point: Point, frame_index: int, used_slots: set[int]) -> int | None:
        best_slot = None
        best_distance = self.match_distance_m

        for slot_id, previous_point in self.slot_positions.items():
            if slot_id in used_slots:
                continue
            missing_frames = frame_index - self.slot_last_seen.get(slot_id, frame_index)
            if missing_frames > self.stale_frames:
                continue
            distance = euclidean_distance(point, previous_point)
            if distance <= best_distance:
                best_distance = distance
                best_slot = slot_id

        return best_slot

    def _new_slot_id(self) -> int:
        while self.next_slot_id in self.slot_positions:
            self.next_slot_id += 1
        slot_id = self.next_slot_id
        self.next_slot_id += 1
        return slot_id

    def _remove_stale_slots(self, frame_index: int) -> None:
        stale_slots = [
            slot_id
            for slot_id, last_seen in self.slot_last_seen.items()
            if frame_index - last_seen > self.stale_frames
        ]
        for slot_id in stale_slots:
            self.slot_positions.pop(slot_id, None)
            self.slot_last_seen.pop(slot_id, None)
            raw_ids = [raw_id for raw_id, mapped_slot in self.raw_to_slot.items() if mapped_slot == slot_id]
            for raw_id in raw_ids:
                self.raw_to_slot.pop(raw_id, None)


def euclidean_distance(a: Point, b: Point) -> float:
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def merge_close_raw_positions(raw_positions: dict[int, Point], merge_distance_m: float) -> dict[int, Point]:
    if merge_distance_m <= 0:
        return dict(raw_positions)

    merged: dict[int, Point] = {}
    for raw_id, point in sorted(raw_positions.items()):
        if any(euclidean_distance(point, kept_point) <= merge_distance_m for kept_point in merged.values()):
            continue
        merged[raw_id] = point
    return merged
