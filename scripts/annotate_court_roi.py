from __future__ import annotations

import argparse
import json
from pathlib import Path


WINDOW_NAME = "Court ROI Annotator"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Click court boundary points on a frame and save configs/court_roi.json."
    )
    parser.add_argument("--image", required=True, help="Input frame image path.")
    parser.add_argument(
        "--output",
        default="configs/court_roi.json",
        help="Output ROI JSON config path.",
    )
    parser.add_argument(
        "--preview",
        default="data/frames/roi_preview.jpg",
        help="Output preview image path.",
    )
    return parser.parse_args()


def write_roi_config(points: list[tuple[int, int]], output_path: str | Path) -> None:
    if len(points) < 3:
        raise ValueError("At least 3 points are required to save a court ROI.")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "court_polygon": [[int(x), int(y)] for x, y in points],
        "anchor": "bottom_center",
    }
    output.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def draw_preview(image_path: str | Path, points: list[tuple[int, int]], preview_path: str | Path) -> None:
    import cv2
    import numpy as np

    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError(f"Could not read image: {image_path}")

    polygon = np.array(points, dtype=np.int32)
    overlay = image.copy()
    cv2.fillPoly(overlay, [polygon], color=(0, 180, 255))
    image = cv2.addWeighted(overlay, 0.20, image, 0.80, 0)
    cv2.polylines(image, [polygon], isClosed=True, color=(0, 180, 255), thickness=3)

    for index, (x, y) in enumerate(points, start=1):
        cv2.circle(image, (x, y), radius=6, color=(0, 0, 255), thickness=-1)
        cv2.putText(
            image,
            str(index),
            (x + 8, y - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

    preview = Path(preview_path)
    preview.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(preview), image):
        raise RuntimeError(f"Could not write preview image: {preview}")


def annotate_roi(image_path: str | Path, output_path: str | Path, preview_path: str | Path) -> None:
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
            cv2.circle(canvas, point, radius=6, color=(0, 0, 255), thickness=-1)
            cv2.putText(
                canvas,
                str(index),
                (point[0] + 8, point[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
        if len(points) >= 2:
            for start, end in zip(points, points[1:]):
                cv2.line(canvas, start, end, color=(0, 180, 255), thickness=2)
        if len(points) >= 3:
            cv2.line(canvas, points[-1], points[0], color=(0, 180, 255), thickness=2)

        instructions = "Click court corners | Enter: save | U: undo | R: reset | Q/Esc: quit"
        cv2.putText(
            canvas,
            instructions,
            (20, 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            (255, 255, 255),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            instructions,
            (20, 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            (20, 20, 20),
            1,
            cv2.LINE_AA,
        )
        cv2.imshow(WINDOW_NAME, canvas)

    def on_mouse(event: int, x: int, y: int, _flags: int, _param: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
            redraw()

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WINDOW_NAME, on_mouse)
    redraw()

    try:
        while True:
            key = cv2.waitKey(20) & 0xFF
            if key in (13, 10):
                write_roi_config(points, output_path)
                draw_preview(image_path, points, preview_path)
                print(f"Saved ROI config to {output_path}")
                print(f"Saved ROI preview to {preview_path}")
                break
            if key in (ord("u"), ord("U")) and points:
                points.pop()
                redraw()
            if key in (ord("r"), ord("R")):
                points.clear()
                redraw()
            if key in (27, ord("q"), ord("Q")):
                print("Canceled. ROI config was not saved.")
                break
    finally:
        cv2.destroyWindow(WINDOW_NAME)


def main() -> None:
    args = parse_args()
    annotate_roi(args.image, args.output, args.preview)


if __name__ == "__main__":
    main()
