from __future__ import annotations

import argparse
from pathlib import Path

try:
    from .pose_utils import POSE_SKELETON_EDGES, visible_pose_points
    from .roi_utils import bottom_center, load_court_polygon, point_in_polygon, polygon_as_int_points
except ImportError:
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
) -> None:
    import cv2
    import numpy as np
    from ultralytics import YOLO

    video = Path(video_path)
    if not video.exists():
        raise FileNotFoundError(f"Video not found: {video}")

    polygon = load_court_polygon(roi_path)
    polygon_points = np.array(polygon_as_int_points(polygon), dtype=np.int32)

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
) -> np.ndarray:
    import numpy as np

    annotated = frame.copy()
    draw_polygon(annotated, polygon_points)

    boxes = getattr(result, "boxes", None)
    if boxes is None or boxes.id is None:
        if not minimal_overlay:
            draw_status(annotated, "No tracked players inside ROI")
        return annotated

    xyxy = boxes.xyxy.cpu().numpy()
    track_ids = boxes.id.cpu().numpy().astype(int)
    confidences = boxes.conf.cpu().numpy() if boxes.conf is not None else np.zeros(len(xyxy))
    pose_xy, pose_scores = get_pose_arrays(result)

    kept_count = 0
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

    if not minimal_overlay:
        draw_status(annotated, f"Players in ROI: {kept_count}")
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
    )
    print(f"Saved tracked video to {args.output}")


if __name__ == "__main__":
    main()
