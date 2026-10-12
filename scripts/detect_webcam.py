from __future__ import annotations

import argparse
import sys
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from vehicle_perception.io import load_labels
from vehicle_perception.model import OpenVINODetector
from vehicle_perception.types import Detection
from vehicle_perception.visualization import draw_detections


WINDOW_NAME = "PP-YOLOE+ Vehicle Perception"


def _window_size(value: str) -> tuple[int, int]:
    try:
        width, height = (int(part) for part in value.lower().split("x", maxsplit=1))
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("window size must look like 960x540") from exc
    if width <= 0 or height <= 0:
        raise argparse.ArgumentTypeError("window dimensions must be positive")
    return width, height


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run real-time PP-YOLOE+ vehicle perception from a camera."
    )
    parser.add_argument("--model", type=Path, required=True, help="OpenVINO model.xml")
    parser.add_argument("--labels", type=Path, required=True, help="One label per line")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--device", default="CPU")
    parser.add_argument("--confidence", type=float, default=0.5)
    parser.add_argument("--iou", type=float, default=0.35)
    parser.add_argument("--window-size", type=_window_size, default=(960, 540))
    return parser


def _submit(
    executor: ThreadPoolExecutor,
    detector: OpenVINODetector,
    frame: np.ndarray,
) -> Future[list[Detection]]:
    return executor.submit(detector.predict, frame)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    capture = None
    try:
        labels = load_labels(args.labels)
        detector = OpenVINODetector(
            args.model,
            labels,
            device=args.device,
            confidence_threshold=args.confidence,
            iou_threshold=args.iou,
            extra_channel_policy="label_unknown",
        )
        capture = cv2.VideoCapture(args.camera)
        if not capture.isOpened():
            print(f"Unable to open camera: {args.camera}", file=sys.stderr)
            return 2

        ok, current_frame = capture.read()
        if not ok or current_frame is None:
            print("Camera did not return an initial frame", file=sys.stderr)
            return 3

        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, *args.window_size)
        with ThreadPoolExecutor(max_workers=2) as executor:
            current_future = _submit(executor, detector, current_frame)
            while capture.isOpened():
                ok, next_frame = capture.read()
                next_future = (
                    _submit(executor, detector, next_frame)
                    if ok and next_frame is not None
                    else None
                )

                detections = current_future.result()
                annotated = draw_detections(current_frame, detections)
                cv2.imshow(WINDOW_NAME, annotated)
                if cv2.waitKey(1) & 0xFF == 27:
                    if next_future is not None:
                        next_future.cancel()
                    break

                if next_future is not None:
                    current_frame, current_future = next_frame, next_future
                    continue

                while capture.isOpened():
                    ok, replacement = capture.read()
                    if ok and replacement is not None:
                        current_frame = replacement
                        current_future = _submit(executor, detector, current_frame)
                        break
                else:
                    break
        return 0
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    finally:
        if capture is not None:
            capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    raise SystemExit(main())
