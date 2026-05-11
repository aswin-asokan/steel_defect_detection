#!/usr/bin/env python3
"""Run MobileSAM exact-boundary segmentation on a single image for testing."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import cv2

os.environ.setdefault("ENABLE_LOCAL_CAMERA", "0")
import backend as service  # noqa: E402


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_IMAGE = BASE_DIR / "sample" / "example1.jpeg"
DEFAULT_OUTPUT_DIR = BASE_DIR / "test_outputs" / "mobilesam"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MobileSAM image segmentation test")
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE, help="Path to input image")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory to save outputs")
    parser.add_argument("--threshold", type=float, default=0.6, help="MobileSAM threshold")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_path = args.image.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    frame = cv2.imread(str(image_path))
    if frame is None:
        raise FileNotFoundError(f"Unable to read image: {image_path}")

    validation = service._ensure_models_for_mode(service.MODE_MOBILE_SAM)
    if not validation["ok"]:
        raise RuntimeError(validation["message"])

    current = service.config_store.get()
    cfg = service.RuntimeConfig(
        mode=service.MODE_MOBILE_SAM,
        yolo_conf=current.yolo_conf,
        sam_threshold=float(args.threshold),
        process_interval=current.process_interval,
    )
    result = service.process_frame(frame, frame_idx=1, cfg=cfg)

    overlay_path = output_dir / f"{image_path.stem}_mobilesam_overlay.jpg"
    mask_path = output_dir / f"{image_path.stem}_mobilesam_mask.png"
    summary_path = output_dir / f"{image_path.stem}_mobilesam_result.json"

    cv2.imwrite(str(overlay_path), result["frame"])
    if result["mask"] is not None:
        cv2.imwrite(str(mask_path), result["mask"])

    summary = {
        "status": "success",
        "mode": cfg.mode,
        "image": str(image_path),
        "overlay_image": str(overlay_path),
        "mask_image": str(mask_path) if result["mask"] is not None else None,
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
