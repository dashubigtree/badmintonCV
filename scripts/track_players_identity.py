"""Track players with persistent P1..P4 identity, ported from the
badminton-tracker repo's ``track_players_identity.py``.

Unlike ``track_players.py`` + ``minimap_identity.py`` (nearest-neighbour slot
matching, no court-side awareness), this script assigns detections through
``identity_manager.IdentityManager``: half-court side constraints + constant-
velocity prediction + raw ByteTrack ID ownership protection. See that module's
docstring for why.

Usage:
    python -m scripts.track_players_identity \\
        --video "data/input/test1.mp4" \\
        --court-calibration configs/test1_court_calibration.json \\
        --output data/output/test1_identity.mp4 \\
        --match-format doubles
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

try:
    from .court_minimap import apply_homography, load_court_calibration, minimap_pixel, point_is_inside_court
    from .identity_manager import IdentityManager, get_ground_point
    from .pose_utils import POSE_SKELETON_EDGES, visible_pose_points
    from .track_players import build_image_to_court_transform, get_pose_arrays
except ImportError:
    from court_minimap import apply_homography, load_court_calibration, minimap_pixel, point_is_inside_court
    from identity_manager import IdentityManager, get_ground_point
    from pose_utils import POSE_SKELETON_EDGES, visible_pose_points
    from track_players import build_image_to_court_transform, get_pose_arrays


PERSON_CLASS_ID = 0

# Fixed per-identity colors (BGR) so P1..P4 stay visually stable across the
# whole video, unlike coloring by the raw (churning) tracker ID.
PLAYER_COLORS = {
    1: (80, 80, 255),
    2: (80, 220, 255),
    3: (80, 255, 80),
    4: (255, 170, 80),
}

CSV_FIELDS = [
    "frame",
    "time_sec",
    "status",
    "player_id",
    "side",
    "raw_track_id",
    "match_mode",
    "match_cost",
    "missing_frames",
    "person_conf",
    "ground_method",
    "ground_quality",
    "ground_status",
    "raw_x_m",
    "raw_y_m",
    "corrected_x_m",
    "corrected_y_m",
    "stable_x_m",
    "stable_y_m",
    "ground_speed_m_per_frame",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Track players with persistent P1-P4 identity (badminton-tracker method)."
    )
    parser.add_argument("--video", required=True, help="Input video path.")
    parser.add_argument("--court-calibration", required=True, help="Court calibration JSON (image_points/court_points_m).")
    parser.add_argument("--output", required=True, help="Output annotated video path.")
    parser.add_argument("--csv", default=None, help="Position CSV output path. Defaults to <output> with .csv suffix.")
    parser.add_argument("--match-format", required=True, choices=["singles", "doubles"], help="Roster constraint.")
    parser.add_argument("--model", default="yolo11s-pose.pt", help="Ultralytics pose model path/name.")
    parser.add_argument("--tracker", default="bytetrack.yaml", help="Ultralytics tracker config.")
    parser.add_argument("--device", default=None, help="Inference device, for example mps or cpu.")
    parser.add_argument("--conf", type=float, default=0.25, help="Detection confidence threshold.")
    parser.add_argument("--imgsz", type=int, default=960, help="YOLO inference image size.")
    parser.add_argument("--pose-conf", type=float, default=0.30, help="Minimum keypoint confidence for drawing pose joints.")
    parser.add_argument("--ankle-conf", type=float, default=0.20, help="Minimum ankle keypoint confidence for ground point.")
    parser.add_argument("--max-frames", type=int, default=None, help="Optional frame limit for quick tests.")
    parser.add_argument("--minimal-overlay", action="store_true", help="Draw only boxes and skeleton, no text labels.")
    parser.add_argument("--minimap-size", type=int, default=220, help="Minimap height in pixels.")
    parser.add_argument("--trail-length", type=int, default=90, help="Number of past stable positions to keep per player.")
    parser.add_argument(
        "--court-margin-m",
        type=float,
        default=1.00,
        help="Extra margin outside the court boundary still considered for identity assignment, in meters.",
    )
    parser.add_argument("--side-tolerance-m", type=float, default=0.65, help="How far past the net a detection may still match its side.")
    parser.add_argument("--max-match-distance-m", type=float, default=3.20, help="Hard cap on match distance regardless of missing frames.")
    parser.add_argument("--max-speed-m-per-frame", type=float, default=0.30, help="Motion bound used for outlier rejection and velocity clipping.")
    parser.add_argument("--identity-timeout-frames", type=int, default=150, help="Frames of absence before a P# is freed up.")
    parser.add_argument("--predict-display-frames", type=int, default=12, help="Frames to keep showing a predicted position during a short occlusion.")
    return parser.parse_args()


def run(args: argparse.Namespace) -> None:
    import cv2
    import numpy as np
    from ultralytics import YOLO

    video = Path(args.video)
    if not video.exists():
        raise FileNotFoundError(f"Video not found: {video}")

    calibration = load_court_calibration(args.court_calibration)
    image_points = np.array(calibration["image_points"], dtype=np.float32)
    court_points = np.array(calibration["court_points_m"], dtype=np.float32)
    homography = build_image_to_court_transform(image_points, court_points)
    inverse_homography = np.linalg.inv(homography)
    court_width_m = calibration["court_size_m"]["width"]
    court_length_m = calibration["court_size_m"]["length"]
    net_y_m = court_length_m / 2.0

    identity_manager = IdentityManager(
        match_format=args.match_format,
        net_y_m=net_y_m,
        side_tolerance_m=args.side_tolerance_m,
        max_match_distance_m=args.max_match_distance_m,
        max_speed_m_per_frame=args.max_speed_m_per_frame,
        identity_timeout_frames=args.identity_timeout_frames,
        predict_display_frames=args.predict_display_frames,
    )

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width <= 0 or height <= 0:
        cap.release()
        raise RuntimeError(f"Could not read video dimensions: {video}")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Could not open output video writer: {output}")

    csv_path = Path(args.csv) if args.csv else output.with_suffix(".csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    csv_file = open(csv_path, "w", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(CSV_FIELDS)

    court_polygon_image = project_court_polygon(inverse_homography, court_width_m, court_length_m)
    model = YOLO(args.model)
    frame_index = 0
    trails: dict[int, list[tuple[float, float]]] = {1: [], 2: [], 3: [], 4: []}

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            if args.max_frames is not None and frame_index >= args.max_frames:
                break

            results = model.track(
                frame,
                persist=True,
                tracker=args.tracker,
                classes=[PERSON_CLASS_ID],
                conf=args.conf,
                imgsz=args.imgsz,
                device=args.device,
                verbose=False,
            )
            result = results[0]

            detections = build_detections(
                result,
                homography,
                court_width_m,
                court_length_m,
                args.court_margin_m,
                args.ankle_conf,
            )

            assignments, predicted = identity_manager.assign(detections, frame_index)

            annotated = frame.copy()
            cv2.polylines(annotated, [court_polygon_image], True, (0, 255, 255), 2, cv2.LINE_AA)

            for item in assignments:
                draw_assignment(annotated, item, args.minimal_overlay, args.pose_conf)
                write_csv_row(csv_writer, frame_index, fps, "detected", identity_manager, item)
                state = identity_manager.states[item["player_id"]]
                if state.stable_x is not None:
                    trails[item["player_id"]].append((state.stable_x, state.stable_y))
                    trails[item["player_id"]] = trails[item["player_id"]][-args.trail_length :]

            for item in predicted:
                write_csv_row(csv_writer, frame_index, fps, "predicted", identity_manager, item)

            if not args.minimal_overlay:
                draw_status(annotated, f"Players: {len(assignments)} / {len(identity_manager.allowed_player_ids)}")

            draw_minimap(
                annotated,
                identity_manager,
                assignments,
                predicted,
                trails,
                court_width_m,
                court_length_m,
                args.minimap_size,
                show_labels=not args.minimal_overlay,
            )

            writer.write(annotated)
            frame_index += 1
    finally:
        cap.release()
        writer.release()
        csv_file.close()

    if frame_index == 0:
        raise RuntimeError(f"No frames were processed from: {video}")


def project_court_polygon(inverse_homography, court_width_m: float, court_length_m: float):
    import numpy as np

    corners_m = [(0.0, 0.0), (court_width_m, 0.0), (court_width_m, court_length_m), (0.0, court_length_m)]
    pixels = [apply_homography(corner, inverse_homography) for corner in corners_m]
    return np.array([[int(round(x)), int(round(y))] for x, y in pixels], dtype=np.int32)


def build_detections(
    result: object,
    homography: object,
    court_width_m: float,
    court_length_m: float,
    court_margin_m: float,
    ankle_conf: float,
) -> list[dict]:
    boxes = getattr(result, "boxes", None)
    if boxes is None or boxes.id is None:
        return []

    xyxy = boxes.xyxy.cpu().numpy()
    track_ids = boxes.id.cpu().numpy().astype(int)
    confidences = boxes.conf.cpu().numpy() if boxes.conf is not None else [0.0] * len(xyxy)
    pose_xy, pose_conf = get_pose_arrays(result)

    detections = []
    for i, (box, track_id, confidence) in enumerate(zip(xyxy, track_ids, confidences)):
        box_tuple = tuple(float(v) for v in box)
        if pose_xy is not None and i < len(pose_xy):
            keypoint_conf = pose_conf[i] if pose_conf is not None else None
            ground = get_ground_point(box_tuple, pose_xy[i], keypoint_conf, ankle_conf=ankle_conf)
        else:
            x1, _y1, x2, y2 = box_tuple
            ground = {"x": (x1 + x2) / 2.0, "y": y2, "method": "bbox_bottom", "left_conf": 0.0, "right_conf": 0.0}

        raw_x, raw_y = apply_homography((ground["x"], ground["y"]), homography)

        inside_analysis_area = (
            -court_margin_m <= raw_x <= court_width_m + court_margin_m
            and -court_margin_m <= raw_y <= court_length_m + court_margin_m
        )
        if not inside_analysis_area:
            continue

        detections.append(
            {
                "uid": i,
                "raw_track_id": int(track_id),
                "box": box_tuple,
                "person_conf": float(confidence),
                "ground_x": ground["x"],
                "ground_y": ground["y"],
                "ground_method": ground["method"],
                "raw_x": raw_x,
                "raw_y": raw_y,
                "pose_xy": pose_xy[i] if pose_xy is not None and i < len(pose_xy) else None,
                "pose_conf": pose_conf[i] if pose_conf is not None and i < len(pose_conf) else None,
            }
        )
    return detections


def draw_assignment(frame, item: dict, minimal_overlay: bool, pose_conf_threshold: float) -> None:
    import cv2

    player_id = item["player_id"]
    detection = item["detection"]
    color = PLAYER_COLORS[player_id]
    x1, y1, x2, y2 = (int(round(v)) for v in detection["box"])
    cv2.rectangle(frame, (x1, y1), (x2, y2), color=color, thickness=2)

    if detection["pose_xy"] is not None:
        points = visible_pose_points(detection["pose_xy"], detection["pose_conf"], pose_conf_threshold)
        for start_index, end_index in POSE_SKELETON_EDGES:
            if start_index in points and end_index in points:
                cv2.line(frame, points[start_index][:2], points[end_index][:2], color=color, thickness=2)
        for _index, (x, y, _label) in points.items():
            cv2.circle(frame, (x, y), radius=4, color=color, thickness=-1)

    if minimal_overlay:
        return

    ground_flag = " !" if item["position_info"]["ground_status"] != "accepted" else ""
    label = f"P{player_id} r{detection['raw_track_id']}{ground_flag}"
    cv2.putText(
        frame,
        label,
        (x1, max(22, y1 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        color,
        2,
        cv2.LINE_AA,
    )


def write_csv_row(csv_writer, frame_index: int, fps: float, status: str, identity_manager: IdentityManager, item: dict) -> None:
    player_id = item["player_id"]
    state = identity_manager.states[player_id]

    if status == "detected":
        detection = item["detection"]
        position_info = item["position_info"]
        csv_writer.writerow(
            [
                frame_index,
                frame_index / fps,
                status,
                player_id,
                state.side,
                detection["raw_track_id"],
                item["match_mode"],
                item["match_cost"],
                state.missing_frames(frame_index),
                detection["person_conf"],
                detection["ground_method"],
                position_info["ground_quality"],
                position_info["ground_status"],
                detection["raw_x"],
                detection["raw_y"],
                position_info["corrected_x"],
                position_info["corrected_y"],
                state.stable_x,
                state.stable_y,
                position_info["ground_speed_m_per_frame"],
            ]
        )
    else:
        csv_writer.writerow(
            [
                frame_index,
                frame_index / fps,
                status,
                player_id,
                state.side,
                "",
                "motion_prediction",
                "",
                item["missing_frames"],
                "",
                "",
                "",
                "motion_prediction",
                "",
                "",
                item["pred_x"],
                item["pred_y"],
                item["pred_x"],
                item["pred_y"],
                "",
            ]
        )


def draw_minimap(
    frame,
    identity_manager: IdentityManager,
    assignments: list[dict],
    predicted: list[dict],
    trails: dict[int, list[tuple[float, float]]],
    court_width_m: float,
    court_length_m: float,
    minimap_height: int,
    show_labels: bool,
) -> None:
    import cv2

    height = max(120, minimap_height)
    width = max(80, int(round(height * court_width_m / court_length_m)))
    padding = max(8, height // 18)
    frame_h, frame_w = frame.shape[:2]
    origin = (max(0, frame_w - width - 18), 18)
    overlay = frame.copy()

    x0, y0 = origin
    x1, y1 = x0 + width, y0 + height
    cv2.rectangle(overlay, (x0, y0), (x1, y1), color=(10, 10, 10), thickness=-1)
    cv2.addWeighted(overlay, 0.48, frame, 0.52, 0, dst=frame)
    cv2.rectangle(frame, (x0, y0), (x1, y1), color=(230, 230, 230), thickness=1)

    def to_pixel(point_m):
        return minimap_pixel(point_m, origin, (width, height), court_width_m, court_length_m, padding)

    line_color = (210, 210, 210)
    cv2.rectangle(frame, to_pixel((0.0, 0.0)), to_pixel((court_width_m, court_length_m)), line_color, 1)
    cv2.line(frame, to_pixel((0.0, court_length_m / 2.0)), to_pixel((court_width_m, court_length_m / 2.0)), line_color, 1)
    cv2.line(frame, to_pixel((court_width_m / 2.0, 0.0)), to_pixel((court_width_m / 2.0, court_length_m)), line_color, 1)

    for player_id, trail in trails.items():
        if player_id not in identity_manager.allowed_player_ids or not trail:
            continue
        color = PLAYER_COLORS[player_id]
        pixel_trail = [to_pixel(point) for point in trail]
        for start, end in zip(pixel_trail, pixel_trail[1:]):
            cv2.line(frame, start, end, color=color, thickness=2)

    for item in assignments:
        player_id = item["player_id"]
        state = identity_manager.states[player_id]
        if state.stable_x is None:
            continue
        color = PLAYER_COLORS[player_id]
        px, py = to_pixel((state.stable_x, state.stable_y))
        cv2.circle(frame, (px, py), radius=5, color=color, thickness=-1)
        cv2.circle(frame, (px, py), radius=7, color=(255, 255, 255), thickness=1)
        if show_labels:
            cv2.putText(frame, f"P{player_id}", (px + 7, py - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 2, cv2.LINE_AA)

    for item in predicted:
        player_id = item["player_id"]
        color = PLAYER_COLORS[player_id]
        held_color = tuple(max(40, channel // 2) for channel in color)
        px, py = to_pixel((item["pred_x"], item["pred_y"]))
        cv2.circle(frame, (px, py), radius=5, color=held_color, thickness=-1)
        cv2.circle(frame, (px, py), radius=7, color=(170, 170, 170), thickness=1)
        if show_labels:
            cv2.putText(frame, f"P{player_id}", (px + 7, py - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (190, 190, 190), 2, cv2.LINE_AA)


def draw_status(frame, text: str) -> None:
    import cv2

    cv2.putText(frame, text, (20, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(frame, text, (20, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (20, 20, 20), 1, cv2.LINE_AA)


def main() -> None:
    args = parse_args()
    run(args)
    print(f"Saved tracked video to {args.output}")


if __name__ == "__main__":
    main()
