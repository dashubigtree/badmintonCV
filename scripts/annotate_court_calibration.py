from __future__ import annotations

import argparse
from pathlib import Path

try:
    from .court_minimap import COURT_CORNER_NAMES, write_court_calibration
except ImportError:
    from court_minimap import COURT_CORNER_NAMES, write_court_calibration


WINDOW_NAME = "Court Calibration Annotator"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Click 4 court corners to map image coordinates to a top-down badminton court."
    )
    parser.add_argument("--image", required=True, help="Input frame image path.")
    parser.add_argument(
        "--output",
        default="configs/court_calibration.json",
        help="Output court calibration JSON config path.",
    )
    parser.add_argument(
        "--preview",
        default="data/frames/court_calibration_preview.jpg",
        help="Output preview image path.",
    )
    return parser.parse_args()


def draw_calibration_preview(image_path: str | Path, points: list[tuple[int, int]], preview_path: str | Path) -> None:
    import cv2
    import numpy as np

    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError(f"Could not read image: {image_path}")

    polygon = np.array(points, dtype=np.int32)
    overlay = image.copy()
    cv2.fillPoly(overlay, [polygon], color=(255, 180, 0))
    image = cv2.addWeighted(overlay, 0.18, image, 0.82, 0)
    cv2.polylines(image, [polygon], isClosed=True, color=(255, 180, 0), thickness=3)

    for index, ((x, y), name) in enumerate(zip(points, COURT_CORNER_NAMES), start=1):
        cv2.circle(image, (x, y), radius=6, color=(0, 0, 255), thickness=-1)
        cv2.putText(
            image,
            f"{index} {name}",
            (x + 8, y - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

    preview = Path(preview_path)
    preview.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(preview), image):
        raise RuntimeError(f"Could not write calibration preview: {preview}")


def annotate_court_calibration(image_path: str | Path, output_path: str | Path, preview_path: str | Path) -> None:
    import cv2

    image_file = Path(image_path)
    if not image_file.exists():
        raise FileNotFoundError(f"Image not found: {image_file}")

    image = cv2.imread(str(image_file))
    if image is None:
        raise RuntimeError(f"Could not read image: {image_file}")

    points: list[tuple[int, int]] = []

    def redraw() -> None:
        canvas = image.copy()
        for index, point in enumerate(points, start=1):
            label = COURT_CORNER_NAMES[index - 1]
            cv2.circle(canvas, point, radius=6, color=(0, 0, 255), thickness=-1)
            cv2.putText(
                canvas,
                f"{index} {label}",
                (point[0] + 8, point[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
        if len(points) >= 2:
            for start, end in zip(points, points[1:]):
                cv2.line(canvas, start, end, color=(255, 180, 0), thickness=2)
        if len(points) == 4:
            cv2.line(canvas, points[-1], points[0], color=(255, 180, 0), thickness=2)

        next_label = COURT_CORNER_NAMES[len(points)] if len(points) < 4 else "press Enter to save"
        instructions = (
            f"Click 4 corners in top-down order: {next_label} | "
            "Enter: save | U: undo | R: reset | Q/Esc: quit"
        )
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
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
            points.append((x, y))
            redraw()

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WINDOW_NAME, on_mouse)
    redraw()

    try:
        while True:
            key = cv2.waitKey(20) & 0xFF
            if key in (13, 10):
                if len(points) != 4:
                    print("Need exactly 4 points before saving.")
                    continue
                write_court_calibration(points, output_path)
                draw_calibration_preview(image_path, points, preview_path)
                print(f"Saved court calibration to {output_path}")
                print(f"Saved calibration preview to {preview_path}")
                break
            if key in (ord("u"), ord("U")) and points:
                points.pop()
                redraw()
            if key in (ord("r"), ord("R")):
                points.clear()
                redraw()
            if key in (27, ord("q"), ord("Q")):
                print("Canceled. Court calibration was not saved.")
                break
    finally:
        cv2.destroyWindow(WINDOW_NAME)


def main() -> None:
    args = parse_args()
    annotate_court_calibration(args.image, args.output, args.preview)


if __name__ == "__main__":
    main()
