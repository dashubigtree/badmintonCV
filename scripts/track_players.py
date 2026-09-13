from __future__ import annotations

import argparse
from collections import defaultdict, deque
from pathlib import Path

try:
    from .court_minimap import (
        apply_homography,
        load_court_calibration,
        minimap_pixel,
        point_is_inside_court,
        stable_track_color,
    )
    from .pose_utils import POSE_SKELETON_EDGES, visible_pose_points
    from .roi_utils import bottom_center, load_court_polygon, point_in_polygon, polygon_as_int_points
except ImportError:
    from court_minimap import (
        apply_homography,
        load_court_calibration,
        minimap_pixel,
        point_is_inside_court,
        stable_track_color,
    )
    from pose_utils import POSE_SKELETON_EDGES, visible_pose_points
    from roi_utils import bottom_center, load_court_polygon, point_in_polygon, polygon_as_int_points


PERSON_CLASS_ID = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Track players in a badminton video and keep only detections inside the court ROI."
    )
    parser.add_argument("--video", required=True, help="Input video path.")
    parser.add_argument("--roi", required=True, help="ROI JSON config path.")
    parser.add_argument("--output", required=True, help="Output annotated video path.")
    parser.add_argument(
        "--model",
        default="yolo11s.pt",
        help="Ultralytics YOLO model path/name. Use a pose model such as yolo11s-pose.pt to draw joints.",
    )
    parser.add_argument("--tracker", default="bytetrack.yaml", help="Ultralytics tracker config.")
    parser.add_argument("--device", default=None, help="Inference device, for example mps or cpu.")
    parser.add_argument("--conf", type=float, default=0.25, help="Detection confidence threshold.")
    parser.add_argument("--imgsz", type=int, default=960, help="YOLO inference image size.")
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional frame limit for quick tests.",
    )
    parser.add_argument(
        "--pose-conf",
        type=float,
        default=0.30,
        help="Minimum keypoint confidence for drawing pose joints.",
    )
    parser.add_argument(
        "--minimal-overlay",
        action="store_true",
        help="Draw only ROI, boxes, skeleton lines, and joint dots without text or numeric values.",
    )
    parser.add_argument(
        "--minimap",
        action="store_true",
        help="Draw a top-down court minimap in the upper-right corner.",
    )
    parser.add_argument(
        "--court-calibration",
        default=None,
        help="Court calibration JSON from annotate_court_calibration.py. Required when --minimap is used.",
    )
    parser.add_argument("--minimap-size", type=int, default=220, help="Minimap height in pixels.")
    parser.add_argument("--trail-length", type=int, default=90, help="Number of past positions to keep per player.")
    return parser.parse_args()


def annotate_video(
    video_path: str | Path,
    roi_path: str | Path,
    output_path: str | Path,
    model_name: str,
    tracker_name: str,
    device: str | None,
    conf: float,
    imgsz: int,
    max_frames: int | None = None,
    pose_conf: float = 0.30,
    minimal_overlay: bool = False,
    minimap: bool = False,
    court_calibration_path: str | Path | None = None,
    minimap_size: int = 220,
    trail_length: int = 90,
) -> None:
    import cv2
    import numpy as np
    from ultralytics import YOLO

    video = Path(video_path)
    if not video.exists():
        raise FileNotFoundError(f"Video not found: {video}")

    polygon = load_court_polygon(roi_path)
    polygon_points = np.array(polygon_as_int_points(polygon), dtype=np.int32)
    minimap_state = None
    if minimap:
        if court_calibration_path is None:
            raise ValueError("--court-calibration is required when --minimap is used.")
        calibration = load_court_calibration(court_calibration_path)
        image_points = np.array(calibration["image_points"], dtype=np.float32)
        court_points = np.array(calibration["court_points_m"], dtype=np.float32)
        homography = cv2.getPerspectiveTransform(image_points, court_points)
        minimap_state = {
            "homography": homography,
            "court_size_m": calibration["court_size_m"],
            "trails": defaultdict(lambda: deque(maxlen=max(1, trail_length))),
            "size": max(120, minimap_size),
        }

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width <= 0 or height <= 0:
        cap.release()
        raise RuntimeError(f"Could not read video dimensions: {video}")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Could not open output video writer: {output}")

    model = YOLO(model_name)
    frame_index = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            if max_frames is not None and frame_index >= max_frames:
                break

            results = model.track(
                frame,
                persist=True,
                tracker=tracker_name,
                classes=[PERSON_CLASS_ID],
                conf=conf,
                imgsz=imgsz,
                device=device,
                verbose=False,
            )
            annotated = annotate_frame(
                frame,
                results[0],
                polygon,
                polygon_points,
                pose_conf,
                minimal_overlay,
                minimap_state,
            )
            writer.write(annotated)
            frame_index += 1
    finally:
        cap.release()
        writer.release()

    if frame_index == 0:
        raise RuntimeError(f"No frames were processed from: {video}")


def annotate_frame(
    frame: np.ndarray,
    result: object,
    polygon: list[tuple[float, float]],
    polygon_points: np.ndarray,
    pose_conf: float,
    minimal_overlay: bool,
    minimap_state: dict[str, object] | None = None,
) -> np.ndarray:
    import numpy as np

    annotated = frame.copy()
    draw_polygon(annotated, polygon_points)

    boxes = getattr(result, "boxes", None)
    if boxes is None or boxes.id is None:
        if not minimal_overlay:
            draw_status(annotated, "No tracked players inside ROI")
        if minimap_state is not None:
            draw_minimap(annotated, {}, minimap_state, show_labels=not minimal_overlay)
        return annotated

    xyxy = boxes.xyxy.cpu().numpy()
    track_ids = boxes.id.cpu().numpy().astype(int)
    confidences = boxes.conf.cpu().numpy() if boxes.conf is not None else np.zeros(len(xyxy))
    pose_xy, pose_scores = get_pose_arrays(result)

    kept_count = 0
    current_positions_m: dict[int, tuple[float, float]] = {}
    for detection_index, (box, track_id, confidence) in enumerate(zip(xyxy, track_ids, confidences)):
        box_tuple = tuple(float(value) for value in box)
        anchor = bottom_center(box_tuple)
        if not point_in_polygon(anchor, polygon):
            continue

        kept_count += 1
        draw_player_box(
            annotated,
            box_tuple,
            track_id,
            float(confidence),
            anchor,
            show_text=not minimal_overlay,
        )
        if pose_xy is not None and detection_index < len(pose_xy):
            keypoint_conf = pose_scores[detection_index] if pose_scores is not None else None
            draw_pose_keypoints(
                annotated,
                pose_xy[detection_index],
                keypoint_conf,
                pose_conf,
                show_labels=not minimal_overlay,
            )
            player_anchor = player_position_from_pose(
                pose_xy[detection_index],
                keypoint_conf,
                fallback=anchor,
                min_confidence=pose_conf,
            )
        else:
            player_anchor = anchor

        if minimap_state is not None:
            court_point = image_point_to_court(player_anchor, minimap_state)
            if court_point is not None:
                current_positions_m[int(track_id)] = court_point

    if not minimal_overlay:
        draw_status(annotated, f"Players in ROI: {kept_count}")
    if minimap_state is not None:
        draw_minimap(annotated, current_positions_m, minimap_state, show_labels=not minimal_overlay)
    return annotated


def get_pose_arrays(result: object) -> tuple[object | None, object | None]:
    keypoints = getattr(result, "keypoints", None)
    if keypoints is None or getattr(keypoints, "xy", None) is None:
        return None, None

    xy = keypoints.xy.cpu().numpy()
    conf = None
    if getattr(keypoints, "conf", None) is not None:
        conf = keypoints.conf.cpu().numpy()
    return xy, conf


def player_position_from_pose(
    xy: object,
    conf: object | None,
    fallback: tuple[float, float],
    min_confidence: float,
) -> tuple[float, float]:
    ankles = []
    for index in (15, 16):
        if index >= len(xy):
            continue
        if conf is not None and index < len(conf) and float(conf[index]) < min_confidence:
            continue
        x, y = xy[index]
        if float(x) <= 0 and float(y) <= 0:
            continue
        ankles.append((float(x), float(y)))

    if ankles:
        avg_x = sum(point[0] for point in ankles) / len(ankles)
        avg_y = sum(point[1] for point in ankles) / len(ankles)
        return (avg_x, avg_y)
    return fallback


def image_point_to_court(
    image_point: tuple[float, float],
    minimap_state: dict[str, object],
) -> tuple[float, float] | None:
    court_size = minimap_state["court_size_m"]
    court_point = apply_homography(image_point, minimap_state["homography"])
    if not point_is_inside_court(court_point, court_size["width"], court_size["length"]):
        return None
    return court_point


def draw_polygon(frame: np.ndarray, polygon_points: np.ndarray) -> None:
    import cv2

    overlay = frame.copy()
    cv2.fillPoly(overlay, [polygon_points], color=(0, 180, 255))
    cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, dst=frame)
    cv2.polylines(frame, [polygon_points], isClosed=True, color=(0, 180, 255), thickness=3)


def draw_player_box(
    frame: np.ndarray,
    box: tuple[float, float, float, float],
    track_id: int,
    confidence: float,
    anchor: tuple[float, float],
    show_text: bool = True,
) -> None:
    import cv2

    x1, y1, x2, y2 = (int(round(value)) for value in box)
    ax, ay = (int(round(value)) for value in anchor)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color=(40, 220, 40), thickness=2)
    cv2.circle(frame, (ax, ay), radius=4, color=(0, 0, 255), thickness=-1)

    if not show_text:
        return

    label = f"ID {track_id} {confidence:.2f}"
    label_y = max(24, y1 - 8)
    cv2.putText(
        frame,
        label,
        (x1, label_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (40, 220, 40),
        2,
        cv2.LINE_AA,
    )


def draw_pose_keypoints(
    frame: np.ndarray,
    xy: object,
    conf: object | None,
    min_confidence: float,
    show_labels: bool = True,
) -> None:
    import cv2

    points = visible_pose_points(xy, conf, min_confidence)
    for start_index, end_index in POSE_SKELETON_EDGES:
        if start_index in points and end_index in points:
            start = points[start_index][:2]
            end = points[end_index][:2]
            cv2.line(frame, start, end, color=(255, 170, 0), thickness=2)

    for _index, (x, y, label) in points.items():
        cv2.circle(frame, (x, y), radius=5, color=(0, 0, 255), thickness=-1)
        cv2.circle(frame, (x, y), radius=7, color=(255, 255, 255), thickness=1)
        if not show_labels:
            continue
        cv2.putText(
            frame,
            label,
            (x + 6, y - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )


def draw_minimap(
    frame: np.ndarray,
    current_positions_m: dict[int, tuple[float, float]],
    minimap_state: dict[str, object],
    show_labels: bool,
) -> None:
    import cv2

    court_size = minimap_state["court_size_m"]
    trails = minimap_state["trails"]
    height = int(minimap_state["size"])
    width = max(80, int(round(height * court_size["width"] / court_size["length"])))
    padding = max(8, height // 18)
    frame_h, frame_w = frame.shape[:2]
    origin = (max(0, frame_w - width - 18), 18)
    overlay = frame.copy()

    x0, y0 = origin
    x1, y1 = x0 + width, y0 + height
    cv2.rectangle(overlay, (x0, y0), (x1, y1), color=(10, 10, 10), thickness=-1)
    cv2.addWeighted(overlay, 0.48, frame, 0.52, 0, dst=frame)
    cv2.rectangle(frame, (x0, y0), (x1, y1), color=(230, 230, 230), thickness=1)

    draw_minimap_court_lines(frame, origin, (width, height), court_size["width"], court_size["length"], padding)

    for track_id, point_m in current_positions_m.items():
        trails[track_id].append(point_m)

    for track_id, trail in list(trails.items()):
        color = stable_track_color(int(track_id))
        pixel_trail = [
            minimap_pixel(point, origin, (width, height), court_size["width"], court_size["length"], padding)
            for point in trail
        ]
        for start, end in zip(pixel_trail, pixel_trail[1:]):
            cv2.line(frame, start, end, color=color, thickness=2)
        if not pixel_trail:
            continue
        px, py = pixel_trail[-1]
        cv2.circle(frame, (px, py), radius=5, color=color, thickness=-1)
        cv2.circle(frame, (px, py), radius=7, color=(255, 255, 255), thickness=1)
        if show_labels:
            cv2.putText(
                frame,
                str(track_id),
                (px + 7, py - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )


def draw_minimap_court_lines(
    frame: np.ndarray,
    origin: tuple[int, int],
    size: tuple[int, int],
    court_width_m: float,
    court_length_m: float,
    padding: int,
) -> None:
    import cv2

    def p(x_m: float, y_m: float) -> tuple[int, int]:
        return minimap_pixel((x_m, y_m), origin, size, court_width_m, court_length_m, padding)

    left_top = p(0.0, 0.0)
    right_top = p(court_width_m, 0.0)
    right_bottom = p(court_width_m, court_length_m)
    left_bottom = p(0.0, court_length_m)
    line_color = (210, 210, 210)
    cv2.rectangle(frame, left_top, right_bottom, color=line_color, thickness=1)
    cv2.line(frame, p(0.0, court_length_m / 2.0), p(court_width_m, court_length_m / 2.0), line_color, 1)
    cv2.line(frame, p(court_width_m / 2.0, 0.0), p(court_width_m / 2.0, court_length_m), line_color, 1)
    for y_m in (1.98, court_length_m - 1.98):
        cv2.line(frame, p(0.0, y_m), p(court_width_m, y_m), line_color, 1)
    cv2.line(frame, left_top, left_bottom, line_color, 1)
    cv2.line(frame, right_top, right_bottom, line_color, 1)


def draw_status(frame: np.ndarray, text: str) -> None:
    import cv2

    cv2.putText(
        frame,
        text,
        (20, 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        3,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        text,
        (20, 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (20, 20, 20),
        1,
        cv2.LINE_AA,
    )


def main() -> None:
    args = parse_args()
    annotate_video(
        video_path=args.video,
        roi_path=args.roi,
        output_path=args.output,
        model_name=args.model,
        tracker_name=args.tracker,
        device=args.device,
        conf=args.conf,
        imgsz=args.imgsz,
        max_frames=args.max_frames,
        pose_conf=args.pose_conf,
        minimal_overlay=args.minimal_overlay,
        minimap=args.minimap,
        court_calibration_path=args.court_calibration,
        minimap_size=args.minimap_size,
        trail_length=args.trail_length,
    )
    print(f"Saved tracked video to {args.output}")


if __name__ == "__main__":
    main()
