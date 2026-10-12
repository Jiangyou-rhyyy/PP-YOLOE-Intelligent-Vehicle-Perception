from __future__ import annotations

import argparse
from pathlib import Path

import cv2
from PIL import Image

from vehicle_perception.io import load_labels
from vehicle_perception.model import OpenVINODetector
from vehicle_perception.visualization import draw_detections


ROOT = Path(__file__).resolve().parents[1]


def _render_demo(
    detector: OpenVINODetector, source: Path, destination: Path
) -> int:
    image = cv2.imread(str(source))
    if image is None:
        raise ValueError(f"Unable to read source image: {source}")
    detections = detector.predict(image)
    annotated = draw_detections(image, detections)
    if not cv2.imwrite(str(destination), annotated, [cv2.IMWRITE_JPEG_QUALITY, 92]):
        raise RuntimeError(f"Unable to write demo image: {destination}")
    return len(detections)


def _build_gif(images: list[Path], destination: Path) -> None:
    frames = [Image.open(path).convert("RGB") for path in images]
    width = max(frame.width for frame in frames)
    height = max(frame.height for frame in frames)
    canvases: list[Image.Image] = []
    for frame in frames:
        canvas = Image.new("RGB", (width, height), "#08111f")
        canvas.paste(frame, ((width - frame.width) // 2, (height - frame.height) // 2))
        canvases.append(canvas)
    canvases[0].save(
        destination,
        save_all=True,
        append_images=canvases[1:],
        duration=1800,
        loop=0,
        optimize=True,
    )
    for frame in frames:
        frame.close()


def _benchmark_svg() -> str:
    return """<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="680" viewBox="0 0 1200 680" role="img" aria-labelledby="title desc">
  <title id="title">PP-YOLOE+ model benchmark comparison</title>
  <desc id="desc">Comparison of mAP, frames per second, and model size for PP-YOLOE+ s, m, and l variants.</desc>
  <defs>
    <linearGradient id="bg" x1="0" y1="0" x2="1" y2="1"><stop stop-color="#071426"/><stop offset="1" stop-color="#102a43"/></linearGradient>
    <linearGradient id="accent" x1="0" y1="0" x2="1" y2="0"><stop stop-color="#30d5c8"/><stop offset="1" stop-color="#3b82f6"/></linearGradient>
    <filter id="shadow"><feDropShadow dx="0" dy="10" stdDeviation="14" flood-opacity=".24"/></filter>
  </defs>
  <rect width="1200" height="680" rx="28" fill="url(#bg)"/>
  <circle cx="1090" cy="80" r="150" fill="#1d4ed8" opacity=".12"/>
  <text x="70" y="80" fill="#f8fafc" font-family="Segoe UI,Arial,sans-serif" font-size="34" font-weight="700">PP-YOLOE+ Model Benchmark</text>
  <text x="70" y="115" fill="#93a4b8" font-family="Segoe UI,Arial,sans-serif" font-size="17">Accuracy, real-time throughput, and deployment footprint</text>
  <g font-family="Segoe UI,Arial,sans-serif">
    <g transform="translate(70 160)" filter="url(#shadow)">
      <rect width="330" height="430" rx="22" fill="#10233b" stroke="#2dd4bf" stroke-width="2"/>
      <rect x="25" y="25" width="92" height="30" rx="15" fill="#134e4a"/><text x="71" y="46" text-anchor="middle" fill="#5eead4" font-size="14" font-weight="700">SELECTED</text>
      <text x="25" y="100" fill="#f8fafc" font-size="27" font-weight="700">PP-YOLOE+_s</text>
      <text x="25" y="145" fill="#94a3b8" font-size="15">mAP@0.5:0.95</text><text x="305" y="145" text-anchor="end" fill="#f8fafc" font-size="24" font-weight="700">40.23%</text>
      <rect x="25" y="165" width="280" height="9" rx="4.5" fill="#1e3a55"/><rect x="25" y="165" width="235" height="9" rx="4.5" fill="url(#accent)"/>
      <text x="25" y="225" fill="#94a3b8" font-size="15">Throughput</text><text x="305" y="225" text-anchor="end" fill="#f8fafc" font-size="24" font-weight="700">17.59 FPS</text>
      <rect x="25" y="245" width="280" height="9" rx="4.5" fill="#1e3a55"/><rect x="25" y="245" width="280" height="9" rx="4.5" fill="#2dd4bf"/>
      <text x="25" y="305" fill="#94a3b8" font-size="15">Model size</text><text x="305" y="305" text-anchor="end" fill="#f8fafc" font-size="24" font-weight="700">2.76 MB</text>
      <text x="25" y="375" fill="#5eead4" font-size="15" font-weight="600">Best real-time balance</text>
    </g>
    <g transform="translate(435 160)" filter="url(#shadow)">
      <rect width="330" height="430" rx="22" fill="#10233b" stroke="#334e68"/>
      <text x="25" y="100" fill="#f8fafc" font-size="27" font-weight="700">PP-YOLOE+_m</text>
      <text x="25" y="145" fill="#94a3b8" font-size="15">mAP@0.5:0.95</text><text x="305" y="145" text-anchor="end" fill="#f8fafc" font-size="24" font-weight="700">45.8%</text>
      <rect x="25" y="165" width="280" height="9" rx="4.5" fill="#1e3a55"/><rect x="25" y="165" width="268" height="9" rx="4.5" fill="#60a5fa"/>
      <text x="25" y="225" fill="#94a3b8" font-size="15">Throughput</text><text x="305" y="225" text-anchor="end" fill="#f8fafc" font-size="24" font-weight="700">14.97 FPS</text>
      <rect x="25" y="245" width="280" height="9" rx="4.5" fill="#1e3a55"/><rect x="25" y="245" width="238" height="9" rx="4.5" fill="#60a5fa"/>
      <text x="25" y="305" fill="#94a3b8" font-size="15">Model size</text><text x="305" y="305" text-anchor="end" fill="#f8fafc" font-size="24" font-weight="700">28.7 MB</text>
      <text x="25" y="375" fill="#93c5fd" font-size="15" font-weight="600">Balanced accuracy profile</text>
    </g>
    <g transform="translate(800 160)" filter="url(#shadow)">
      <rect width="330" height="430" rx="22" fill="#10233b" stroke="#334e68"/>
      <text x="25" y="100" fill="#f8fafc" font-size="27" font-weight="700">PP-YOLOE+_l</text>
      <text x="25" y="145" fill="#94a3b8" font-size="15">mAP@0.5:0.95</text><text x="305" y="145" text-anchor="end" fill="#f8fafc" font-size="24" font-weight="700">47.9%</text>
      <rect x="25" y="165" width="280" height="9" rx="4.5" fill="#1e3a55"/><rect x="25" y="165" width="280" height="9" rx="4.5" fill="#a78bfa"/>
      <text x="25" y="225" fill="#94a3b8" font-size="15">Throughput</text><text x="305" y="225" text-anchor="end" fill="#f8fafc" font-size="24" font-weight="700">10.88 FPS</text>
      <rect x="25" y="245" width="280" height="9" rx="4.5" fill="#1e3a55"/><rect x="25" y="245" width="173" height="9" rx="4.5" fill="#a78bfa"/>
      <text x="25" y="305" fill="#94a3b8" font-size="15">Model size</text><text x="305" y="305" text-anchor="end" fill="#f8fafc" font-size="24" font-weight="700">196 MB</text>
      <text x="25" y="375" fill="#c4b5fd" font-size="15" font-weight="600">Highest measured accuracy</text>
    </g>
  </g>
  <text x="70" y="640" fill="#64748b" font-family="Segoe UI,Arial,sans-serif" font-size="14">Project test results · PP-YOLOE+_s selected for deployment</text>
</svg>
"""


def build_assets(
    model_path: Path,
    labels_path: Path,
    source_dir: Path,
    output_dir: Path,
) -> dict[str, int]:
    output_dir.mkdir(parents=True, exist_ok=True)
    detector = OpenVINODetector(
        model_path,
        load_labels(labels_path),
        extra_channel_policy="label_unknown",
    )
    counts = {
        "day": _render_demo(detector, source_dir / "day.jpg", output_dir / "demo-day.jpg"),
        "night": _render_demo(
            detector, source_dir / "night.jpg", output_dir / "demo-night.jpg"
        ),
    }
    _build_gif(
        [output_dir / "demo-day.jpg", output_dir / "demo-night.jpg"],
        output_dir / "demo.gif",
    )
    (output_dir / "benchmark.svg").write_text(_benchmark_svg(), encoding="utf-8")
    return counts


def main() -> int:
    parser = argparse.ArgumentParser(description="Rebuild README demo assets")
    parser.add_argument("--model", type=Path, default=ROOT / "model" / "model.xml")
    parser.add_argument("--labels", type=Path, default=ROOT / "model" / "labels.txt")
    parser.add_argument("--source-dir", type=Path, default=ROOT / "assets" / "source")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "assets")
    args = parser.parse_args()
    counts = build_assets(
        args.model.resolve(),
        args.labels.resolve(),
        args.source_dir.resolve(),
        args.output_dir.resolve(),
    )
    print(f"Generated day/night demos with {counts['day']} and {counts['night']} detections")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
