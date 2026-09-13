from __future__ import annotations

import argparse
from pathlib import Path

from roi_utils import load_court_polygon, polygon_as_int_points


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Draw a court ROI preview on an image.")
    parser.add_argument("--image", required=True, help="Input frame image path.")
    parser.add_argument("--roi", required=True, help="ROI JSON config path.")
    parser.add_argument("--output", required=True, help="Output preview image path.")
    return parser.parse_args()


def draw_roi_preview(image_path: str | Path, roi_path: str | Path, output_path: str | Path) -> None:
    import cv2
    import numpy as np

    image_file = Path(image_path)
    if not image_file.exists():
        raise FileNotFoundError(f"Image not found: {image_file}")

    image = cv2.imread(str(image_file))
    if image is None:
        raise RuntimeError(f"Could not read image: {image_file}")

    polygon = load_court_polygon(roi_path)
    points = np.array(polygon_as_int_points(polygon), dtype=np.int32)

    overlay = image.copy()
    cv2.fillPoly(overlay, [points], color=(0, 180, 255))
    image = cv2.addWeighted(overlay, 0.20, image, 0.80, 0)
    cv2.polylines(image, [points], isClosed=True, color=(0, 180, 255), thickness=3)

    for index, (x, y) in enumerate(points.tolist(), start=1):
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

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), image):
        raise RuntimeError(f"Could not write ROI preview: {output}")


def main() -> None:
    args = parse_args()
    draw_roi_preview(args.image, args.roi, args.output)
    print(f"Saved ROI preview to {args.output}")


if __name__ == "__main__":
    main()
