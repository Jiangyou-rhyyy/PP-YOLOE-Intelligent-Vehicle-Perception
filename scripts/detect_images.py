from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2

from vehicle_perception.io import load_labels, write_competition_json
from vehicle_perception.model import OpenVINODetector
from vehicle_perception.visualization import draw_detections


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run PP-YOLOE+ vehicle perception on an image or directory."
    )
    parser.add_argument("--model", type=Path, required=True, help="OpenVINO model.xml")
    parser.add_argument("--labels", type=Path, required=True, help="One label per line")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--image", type=Path, help="Single input image")
    source.add_argument("--image-dir", type=Path, help="Directory of input images")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/images"))
    parser.add_argument("--json-dir", type=Path, help="Optional competition JSON output")
    parser.add_argument("--device", default="CPU")
    parser.add_argument("--confidence", type=float, default=0.5)
    parser.add_argument("--iou", type=float, default=0.35)
    return parser


def _input_paths(image: Path | None, image_dir: Path | None) -> list[Path]:
    if image is not None:
        path = image.expanduser().resolve()
        if not path.is_file():
            raise ValueError(f"Input image not found: {path}")
        return [path]

    assert image_dir is not None
    directory = image_dir.expanduser().resolve()
    if not directory.is_dir():
        raise ValueError(f"Input directory not found: {directory}")
    paths = sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not paths:
        raise ValueError(f"No supported images found in: {directory}")
    return paths


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        paths = _input_paths(args.image, args.image_dir)
        labels = load_labels(args.labels)
        detector = OpenVINODetector(
            args.model,
            labels,
            device=args.device,
            confidence_threshold=args.confidence,
            iou_threshold=args.iou,
            extra_channel_policy="label_unknown",
        )
        output_dir = args.output_dir.expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        json_dir = args.json_dir.expanduser().resolve() if args.json_dir else None
        if json_dir is not None:
            json_dir.mkdir(parents=True, exist_ok=True)

        for image_path in paths:
            image = cv2.imread(str(image_path))
            if image is None:
                print(f"Unable to read image: {image_path}", file=sys.stderr)
                return 2
            detections = detector.predict(image)
            annotated = draw_detections(image, detections)
            destination = output_dir / image_path.name
            if not cv2.imwrite(str(destination), annotated):
                print(f"Unable to write image: {destination}", file=sys.stderr)
                return 3
            if json_dir is not None:
                write_competition_json(
                    json_dir / f"{image_path.stem}.json",
                    image_name=image_path.stem,
                    detections=detections,
                    timestamp=int(time.time()),
                )
            print(f"{image_path.name}: {len(detections)} detections -> {destination}")
        return 0
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
