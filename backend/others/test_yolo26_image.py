#!/usr/bin/env python3
"""Run YOLO26 bounding-box detection on a single image for testing."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import cv2

os.environ.setdefault("ENABLE_LOCAL_CAMERA", "0")
import backend as service  # noqa: E402


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_IMAGE = BASE_DIR / "sample" / "In_4.bmp"
DEFAULT_OUTPUT_DIR = BASE_DIR / "test_outputs" / "yolo26"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLO26 image detection test")
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE, help="Path to input image")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory to save outputs")
    parser.add_argument("--conf", type=float, default=0.25, help="YOLO confidence threshold")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_path = args.image.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    frame = cv2.imread(str(image_path))
    if frame is None:
        raise FileNotFoundError(f"Unable to read image: {image_path}")

    validation = service._ensure_models_for_mode(service.MODE_YOLO26)
    if not validation["ok"]:
        raise RuntimeError(validation["message"])

    cfg = service.RuntimeConfig(
        mode=service.MODE_YOLO26,
        yolo_conf=0.05,
        sam_threshold=service.config_store.get().sam_threshold,
        process_interval=service.config_store.get().process_interval,
    )
    result = service.process_frame(frame, frame_idx=1, cfg=cfg)

    overlay_path = output_dir / f"{image_path.stem}_yolo26_overlay.jpg"
    summary_path = output_dir / f"{image_path.stem}_yolo26_result.json"

    cv2.imwrite(str(overlay_path), result["frame"])

    summary = {
        "status": "success",
        "mode": cfg.mode,
        "image": str(image_path),
        "overlay_image": str(overlay_path),
        "defect": bool(result["defect"]),
        "defect_types": result["defect_types"],
        "detections": result["detections"],
        "contours": int(result["contours"]),
        "sam_confidence": float(result["sam_confidence"]),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
