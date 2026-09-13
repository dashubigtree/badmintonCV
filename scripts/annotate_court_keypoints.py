from __future__ import annotations

import argparse
from pathlib import Path

try:
    from .court_minimap import (
        BEST_FOUR_FAR_NET_LABELS,
        THREE_POINT_FAR_LABELS,
        write_best_four_far_net_calibration,
        write_three_point_far_calibration,
    )
except ImportError:
    from court_minimap import (
        BEST_FOUR_FAR_NET_LABELS,
        THREE_POINT_FAR_LABELS,
        write_best_four_far_net_calibration,
        write_three_point_far_calibration,
    )


WINDOW_NAME = "Court Keypoint Calibration Annotator"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Click visible court landmarks for cropped videos. Default mode uses the recommended "
            "4 points: far doubles long service left/right, net right, net left."
        )
    )
    parser.add_argument("--image", required=True, help="Input frame image path.")
    parser.add_argument(
        "--output",
        default="configs/court_calibration.json",
        help="Output court calibration JSON config path.",
    )
    parser.add_argument(
        "--preview",
        default="data/frames/court_keypoint_calibration_preview.jpg",
        help="Output preview image path.",
    )
    parser.add_argument(
        "--mode",
        choices=("best-four", "three-point-far"),
        default="best-four",
        help="Calibration point set. Default: best-four.",
    )
    return parser.parse_args()


def labels_for_mode(mode: str) -> tuple[str, ...]:
    if mode == "three-point-far":
        return THREE_POINT_FAR_LABELS
    return BEST_FOUR_FAR_NET_LABELS


def save_calibration(points: list[tuple[int, int]], output_path: str | Path, mode: str) -> None:
    if mode == "three-point-far":
        write_three_point_far_calibration(points, output_path)
        return
    write_best_four_far_net_calibration(points, output_path)


def draw_keypoint_preview(
    image_path: str | Path,
    points: list[tuple[int, int]],
    preview_path: str | Path,
    labels: tuple[str, ...],
) -> None:
    import cv2

    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError(f"Could not read image: {image_path}")

    for index, ((x, y), name) in enumerate(zip(points, labels), start=1):
        cv2.circle(image, (x, y), radius=7, color=(0, 0, 255), thickness=-1)
        cv2.putText(
            image,
            f"{index} {name}",
            (x + 8, y - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

    if len(points) >= 2:
        for start, end in zip(points, points[1:]):
            cv2.line(image, start, end, color=(255, 180, 0), thickness=2)

    preview = Path(preview_path)
    preview.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(preview), image):
        raise RuntimeError(f"Could not write keypoint calibration preview: {preview}")


def annotate_court_keypoints(
    image_path: str | Path,
    output_path: str | Path,
    preview_path: str | Path,
    mode: str,
) -> None:
    import cv2

    image_file = Path(image_path)
    if not image_file.exists():
        raise FileNotFoundError(f"Image not found: {image_file}")

    image = cv2.imread(str(image_file))
    if image is None:
        raise RuntimeError(f"Could not read image: {image_file}")

    labels = labels_for_mode(mode)
    points: list[tuple[int, int]] = []

    def redraw() -> None:
        canvas = image.copy()
        for index, point in enumerate(points, start=1):
            label = labels[index - 1]
            cv2.circle(canvas, point, radius=7, color=(0, 0, 255), thickness=-1)
            cv2.putText(
                canvas,
                f"{index} {label}",
                (point[0] + 8, point[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
        if len(points) >= 2:
            for start, end in zip(points, points[1:]):
                cv2.line(canvas, start, end, color=(255, 180, 0), thickness=2)

        next_label = labels[len(points)] if len(points) < len(labels) else "press Enter to save"
        instructions = f"Click: {next_label} | Enter: save | U: undo | R: reset | Q/Esc: quit"
        cv2.putText(
            canvas,
            instructions,
            (20, 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.78,
            (255, 255, 255),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            instructions,
            (20, 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.78,
            (20, 20, 20),
            1,
            cv2.LINE_AA,
        )
        cv2.imshow(WINDOW_NAME, canvas)

    def on_mouse(event: int, x: int, y: int, _flags: int, _param: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < len(labels):
            points.append((x, y))
            redraw()

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WINDOW_NAME, on_mouse)
    redraw()

    try:
        while True:
            key = cv2.waitKey(20) & 0xFF
            if key in (13, 10):
                if len(points) != len(labels):
                    print(f"Need exactly {len(labels)} points before saving.")
                    continue
                save_calibration(points, output_path, mode)
                draw_keypoint_preview(image_path, points, preview_path, labels)
                print(f"Saved court keypoint calibration to {output_path}")
                print(f"Saved keypoint calibration preview to {preview_path}")
                break
            if key in (ord("u"), ord("U")) and points:
                points.pop()
                redraw()
            if key in (ord("r"), ord("R")):
                points.clear()
                redraw()
            if key in (27, ord("q"), ord("Q")):
                print("Canceled. Court keypoint calibration was not saved.")
                break
    finally:
        cv2.destroyWindow(WINDOW_NAME)


def main() -> None:
    args = parse_args()
    annotate_court_keypoints(args.image, args.output, args.preview, args.mode)


if __name__ == "__main__":
    main()
