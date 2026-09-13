from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract one frame from a badminton video for ROI annotation."
    )
    parser.add_argument("--video", required=True, help="Input video path.")
    parser.add_argument("--output", required=True, help="Output image path, usually first_frame.jpg.")
    parser.add_argument(
        "--frame-index",
        type=int,
        default=0,
        help="Zero-based frame index to extract. Defaults to the first frame.",
    )
    return parser.parse_args()


def extract_frame(video_path: str | Path, output_path: str | Path, frame_index: int = 0) -> None:
    import cv2

    if frame_index < 0:
        raise ValueError("--frame-index must be 0 or greater.")

    video = Path(video_path)
    if not video.exists():
        raise FileNotFoundError(f"Video not found: {video}")

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video}")

    try:
        if frame_index:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = cap.read()
    finally:
        cap.release()

    if not ok or frame is None:
        raise RuntimeError(f"Could not read frame {frame_index} from: {video}")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), frame):
        raise RuntimeError(f"Could not write output image: {output}")


def main() -> None:
    args = parse_args()
    extract_frame(args.video, args.output, args.frame_index)
    print(f"Saved frame {args.frame_index} to {args.output}")


if __name__ == "__main__":
    main()
