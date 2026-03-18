#!/usr/bin/env python3
"""Unified Flask backend for SAM, MobileSAM, YOLOv26, and combined pipelines."""

from __future__ import annotations

import base64
import csv
import json
import sys
import threading
import time
import traceback
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from flask import Flask, Response, jsonify, request
from flask_cors import CORS

from pattern_service import analyze_detection_csv

BASE_DIR = Path(__file__).resolve().parent
LOG_DIR = BASE_DIR / "logs"
MOBILE_SAM_REPO = BASE_DIR / "MobileSAM"
if MOBILE_SAM_REPO.exists() and str(MOBILE_SAM_REPO) not in sys.path:
    sys.path.insert(0, str(MOBILE_SAM_REPO))

MODE_SAM = "sam"
MODE_MOBILE_SAM = "mobilesam"
MODE_YOLO26 = "yolo26"
MODE_YOLO26_MOBILE_SAM = "yolo26_mobilesam"
SUPPORTED_MODES = {
    MODE_SAM,
    MODE_MOBILE_SAM,
    MODE_YOLO26,
    MODE_YOLO26_MOBILE_SAM,
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
STD = np.array([58.395, 57.12, 57.375], dtype=np.float32)

# User-facing FE option names mapped to internal mode + preprocess flag.
SWITCH_OPTIONS: dict[str, dict[str, Any]] = {
    "sam": {"mode": MODE_SAM, "use_preprocess": False},
    "mobilesam": {"mode": MODE_MOBILE_SAM, "use_preprocess": False},
    "yolo26": {"mode": MODE_YOLO26, "use_preprocess": False},
    "yolo26_mobilesam": {"mode": MODE_YOLO26_MOBILE_SAM, "use_preprocess": False},
    "preprocess_yolo26_mobilesam": {"mode": MODE_YOLO26_MOBILE_SAM, "use_preprocess": True},
}


@dataclass
class ModelPaths:
    yolo_candidates: list[Path]
    mobile_sam_checkpoint_candidates: list[Path]
    mobile_sam_weights_candidates: list[Path]
    sam_lora_candidates: list[Path]


def _first_existing(paths: list[Path]) -> Path | None:
    return next((p for p in paths if p.exists()), None)


def get_model_paths() -> ModelPaths:
    # Keep compatibility with both old and moved directory layouts.
    yolo_candidates = [
        BASE_DIR / "runs" / "detect" / "train5" / "weights" / "best.pt",  # old combined.py relative layout
        BASE_DIR / "yolo26" / "runs" / "detect" / "train5" / "weights" / "best.pt",  # moved layout
        BASE_DIR / "yolo26" / "runs" / "yolo26s_ppy" / "exp_light" / "weights" / "best.pt",
        BASE_DIR / "yolo26" / "yolo26s.pt",
    ]
    mobile_sam_checkpoint_candidates = [
        BASE_DIR / "mobile_sam.pt",
        BASE_DIR / "MobileSAM" / "weights" / "mobile_sam.pt",
    ]
    mobile_sam_weights_candidates = [
        BASE_DIR / "mobilesam_defect_optimized" / "best_model.pth",
        BASE_DIR / "mobilesam_defect_optimized" / "final_model.pth",
    ]
    sam_lora_candidates = [
        BASE_DIR / "sam_steel_lora",
        BASE_DIR / "sam_steel_lora" / "checkpoint_epoch_6",
        BASE_DIR / "sam_steel_lora" / "checkpoint_epoch_4",
        BASE_DIR / "sam_steel_lora" / "checkpoint_epoch_2",
    ]
    return ModelPaths(
        yolo_candidates=yolo_candidates,
        mobile_sam_checkpoint_candidates=mobile_sam_checkpoint_candidates,
        mobile_sam_weights_candidates=mobile_sam_weights_candidates,
        sam_lora_candidates=sam_lora_candidates,
    )


@dataclass
class RuntimeConfig:
    mode: str = MODE_YOLO26_MOBILE_SAM
    use_preprocess: bool = False
    process_interval: float = 0.08
    yolo_conf: float = 0.25
    sam_threshold: float = 0.6


class ConfigStore:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._cfg = RuntimeConfig()

    def get(self) -> RuntimeConfig:
        with self._lock:
            return RuntimeConfig(**self._cfg.__dict__)

    def update(self, payload: dict[str, Any]) -> RuntimeConfig:
        with self._lock:
            if "mode" in payload:
                mode = str(payload["mode"]).strip().lower()
                if mode not in SUPPORTED_MODES:
                    raise ValueError(f"Unsupported mode: {mode}")
                self._cfg.mode = mode
            if "use_preprocess" in payload:
                self._cfg.use_preprocess = bool(payload["use_preprocess"])
            if "yolo_conf" in payload:
                self._cfg.yolo_conf = float(payload["yolo_conf"])
            if "sam_threshold" in payload:
                self._cfg.sam_threshold = float(payload["sam_threshold"])
            if "process_interval" in payload:
                self._cfg.process_interval = max(0.02, float(payload["process_interval"]))
            return RuntimeConfig(**self._cfg.__dict__)


class CombinedLogger:
    """Logs detections for yolo26 + mobilesam pipeline only."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.raw_dir = root / "raw"
        self.overlay_dir = root / "overlay"
        self.mask_dir = root / "mask"
        self.pattern_dir = root / "pattern"
        for path in (self.root, self.raw_dir, self.overlay_dir, self.mask_dir, self.pattern_dir):
            path.mkdir(parents=True, exist_ok=True)

        self._lock = threading.Lock()
        self.session_start = datetime.now()
        stamp = self.session_start.strftime("%Y%m%d_%H%M%S")
        self.csv_path = self.root / f"session_{stamp}.csv"
        self.json_path = self.root / f"detections_{stamp}.json"
        self.records: list[dict[str, Any]] = []
        self._init_csv()

    def _init_csv(self) -> None:
        with self.csv_path.open("w", newline="", encoding="utf-8") as fp:
            writer = csv.writer(fp)
            writer.writerow(
                [
                    "timestamp",
                    "frame_index",
                    "raw_image",
                    "overlay_image",
                    "mask_image",
                    "sam_confidence",
                    "defect_area_px",
                    "yolo_labels",
                    "yolo_confs",
                    "yolo_bboxes",
                ]
            )

    def log_detection(
        self,
        frame_idx: int,
        frame: np.ndarray,
        overlay: np.ndarray,
        mask: np.ndarray,
        sam_confidence: float,
        defect_area_px: float,
        detections: list[dict[str, Any]],
    ) -> dict[str, Any]:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        raw_path = self.raw_dir / f"{ts}.jpg"
        overlay_path = self.overlay_dir / f"{ts}.jpg"
        mask_path = self.mask_dir / f"{ts}.png"

        cv2.imwrite(str(raw_path), frame)
        cv2.imwrite(str(overlay_path), overlay)
        cv2.imwrite(str(mask_path), mask)

        labels = [str(d.get("type", "unknown")) for d in detections]
        confs = [f"{float(d.get('confidence', 0.0)):.3f}" for d in detections]
        boxes = []
        for d in detections:
            x1, y1, x2, y2 = d.get("bbox", [0, 0, 0, 0])
            boxes.append(f"{int(x1)},{int(y1)},{int(x2)},{int(y2)}")

        entry = {
            "timestamp": ts,
            "frame_index": int(frame_idx),
            "raw_image": str(raw_path),
            "overlay_image": str(overlay_path),
            "mask_image": str(mask_path),
            "sam_confidence": float(sam_confidence),
            "defect_area_px": float(defect_area_px),
            "yolo_labels": labels,
            "yolo_confs": confs,
            "yolo_bboxes": boxes,
        }

        with self._lock:
            with self.csv_path.open("a", newline="", encoding="utf-8") as fp:
                writer = csv.writer(fp)
                writer.writerow(
                    [
                        entry["timestamp"],
                        entry["frame_index"],
                        entry["raw_image"],
                        entry["overlay_image"],
                        entry["mask_image"],
                        entry["sam_confidence"],
                        entry["defect_area_px"],
                        ";".join(entry["yolo_labels"]),
                        ";".join(entry["yolo_confs"]),
                        ";".join(entry["yolo_bboxes"]),
                    ]
                )
            self.records.append(entry)
            with self.json_path.open("w", encoding="utf-8") as fp:
                json.dump(
                    {
                        "session_start": self.session_start.isoformat(),
                        "last_updated": datetime.now().isoformat(),
                        "total_detections": len(self.records),
                        "detections": self.records,
                    },
                    fp,
                    indent=2,
                )

        return entry


class ModelHub:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.paths = get_model_paths()
        self.yolo = None
        self.yolo_names: dict[int, str] = {}
        self.yolo_model_path: Path | None = None
        self.mobile_sam = None
        self.mobile_sam_checkpoint_path: Path | None = None
        self.mobile_sam_weights_path: Path | None = None
        self.sam_model = None
        self.sam_processor = None
        self.sam_lora_path: Path | None = None

    def get_yolo(self):
        with self._lock:
            if self.yolo is not None:
                return self.yolo, self.yolo_names

            from ultralytics import YOLO  # lazy import
            model_path = _first_existing(self.paths.yolo_candidates)
            if model_path is None:
                raise FileNotFoundError("No YOLO weights found in backend/yolo26")

            self.yolo = YOLO(str(model_path))
            self.yolo_model_path = model_path
            names = getattr(self.yolo.model, "names", {})
            if isinstance(names, list):
                self.yolo_names = {idx: name for idx, name in enumerate(names)}
            elif isinstance(names, dict):
                self.yolo_names = {int(k): str(v) for k, v in names.items()}
            else:
                self.yolo_names = {}
            return self.yolo, self.yolo_names

    def get_mobile_sam(self):
        with self._lock:
            if self.mobile_sam is not None:
                return self.mobile_sam

            from mobile_sam import sam_model_registry  # lazy import
            checkpoint_path = _first_existing(self.paths.mobile_sam_checkpoint_candidates)
            if checkpoint_path is None or not checkpoint_path.exists():
                raise FileNotFoundError(f"Missing MobileSAM checkpoint: {checkpoint_path}")
            finetuned_path = _first_existing(self.paths.mobile_sam_weights_candidates)
            if finetuned_path is None:
                raise FileNotFoundError("Missing MobileSAM fine-tuned weights")

            sam = sam_model_registry["vit_t"](checkpoint=str(checkpoint_path))
            try:
                checkpoint = torch.load(str(finetuned_path), map_location=DEVICE, weights_only=False)
            except TypeError:
                checkpoint = torch.load(str(finetuned_path), map_location=DEVICE)
            state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
            sam.load_state_dict(state_dict)
            sam.to(DEVICE)
            sam.eval()
            self.mobile_sam = sam
            self.mobile_sam_checkpoint_path = checkpoint_path
            self.mobile_sam_weights_path = finetuned_path
            return self.mobile_sam

    def get_sam_lora(self):
        with self._lock:
            if self.sam_model is not None and self.sam_processor is not None:
                return self.sam_model, self.sam_processor

            from peft import PeftModel  # lazy import
            from transformers import SamModel, SamProcessor

            model_dir = _first_existing(self.paths.sam_lora_candidates)
            if model_dir is None or not model_dir.exists():
                raise FileNotFoundError(f"Missing SAM LoRA directory: {model_dir}")

            processor = SamProcessor.from_pretrained(str(model_dir))
            base_model = SamModel.from_pretrained("facebook/sam-vit-base")
            model = PeftModel.from_pretrained(base_model, str(model_dir))
            model.to(DEVICE)
            model.eval()

            self.sam_model = model
            self.sam_processor = processor
            self.sam_lora_path = model_dir
            return self.sam_model, self.sam_processor


def _encode_frame_b64(frame_bgr: np.ndarray) -> str:
    ok, jpeg = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    if not ok:
        raise RuntimeError("Failed to JPEG encode frame")
    b64 = base64.b64encode(jpeg.tobytes()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def _normalize_rgb(img_rgb: np.ndarray) -> np.ndarray:
    return (img_rgb.astype(np.float32) - MEAN) / STD


def _ensure_odd(k: int) -> int:
    return k if k % 2 else k + 1


def preprocess_frame(frame: np.ndarray) -> np.ndarray:
    """Preprocess video frame to look like NEU-DET grayscale-style training images."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    bg_k = _ensure_odd(max(3, min(51, min(frame.shape[:2]) - 1)))
    background = cv2.GaussianBlur(gray, (bg_k, bg_k), 0)
    corrected = cv2.divide(gray, background, scale=255.0)
    corrected = np.clip(corrected, 0, 255).astype(np.uint8)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(corrected)
    enhanced = cv2.GaussianBlur(enhanced, (3, 3), 0)

    bright_mask = (enhanced >= 240).astype(np.uint8) * 255
    if np.any(bright_mask):
        bright_mask = cv2.dilate(bright_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
        try:
            enhanced = cv2.inpaint(enhanced, bright_mask, 3, cv2.INPAINT_TELEA)
        except cv2.error:
            pass

    enhanced = cv2.normalize(enhanced, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)


def _extract_yolo_boxes(result_obj, names: dict[int, str]) -> list[dict[str, Any]]:
    detections: list[dict[str, Any]] = []
    if not hasattr(result_obj, "boxes") or len(result_obj.boxes) == 0:
        return detections

    boxes = result_obj.boxes
    xyxy = boxes.xyxy.detach().cpu().numpy()
    conf = boxes.conf.detach().cpu().numpy() if boxes.conf is not None else np.ones((len(xyxy),), dtype=np.float32)
    cls = boxes.cls.detach().cpu().numpy() if boxes.cls is not None else np.full((len(xyxy),), -1)

    for idx in range(len(xyxy)):
        x1, y1, x2, y2 = [int(v) for v in xyxy[idx]]
        cls_id = int(cls[idx]) if idx < len(cls) else -1
        label = names.get(cls_id, str(cls_id))
        detections.append(
            {
                "type": label,
                "confidence": float(conf[idx]) if idx < len(conf) else 0.0,
                "class_id": cls_id,
                "bbox": [x1, y1, x2, y2],
            }
        )
    return detections


def run_yolo(frame_bgr: np.ndarray, conf: float) -> list[dict[str, Any]]:
    yolo_model, names = model_hub.get_yolo()
    results = yolo_model.predict(frame_bgr, conf=conf, imgsz=512, verbose=False)
    return _extract_yolo_boxes(results[0], names)


def run_mobilesam_on_roi(frame_bgr: np.ndarray, bbox: list[int], threshold: float) -> tuple[np.ndarray, float]:
    sam = model_hub.get_mobile_sam()
    x1, y1, x2, y2 = bbox
    h, w = frame_bgr.shape[:2]
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(1, min(x2, w))
    y2 = max(1, min(y2, h))

    roi = frame_bgr[y1:y2, x1:x2]
    if roi.size == 0:
        return np.zeros((h, w), dtype=np.uint8), 0.0

    roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    sam_size = 1024
    roi_resized = cv2.resize(roi_rgb, (sam_size, sam_size))
    roi_norm = _normalize_rgb(roi_resized)
    tensor = torch.from_numpy(roi_norm).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE)

    with torch.no_grad():
        embedding = sam.image_encoder(tensor)
        sparse, dense = sam.prompt_encoder(points=None, boxes=None, masks=None)
        low_res, _ = sam.mask_decoder(
            image_embeddings=embedding,
            image_pe=sam.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse,
            dense_prompt_embeddings=dense,
            multimask_output=False,
        )
        pred = F.interpolate(low_res, size=(sam_size, sam_size), mode="bilinear", align_corners=False)
        prob = torch.sigmoid(pred)[0, 0].detach().cpu().numpy()

    mask_small = (prob > threshold).astype(np.uint8) * 255
    mask_roi = cv2.resize(mask_small, (roi.shape[1], roi.shape[0]), interpolation=cv2.INTER_NEAREST)

    mask_full = np.zeros((h, w), dtype=np.uint8)
    mask_full[y1:y2, x1:x2] = mask_roi

    active = prob[prob > threshold]
    confidence = float(active.mean()) if active.size else 0.0
    return mask_full, confidence


def run_sam_lora(frame_bgr: np.ndarray, threshold: float) -> tuple[np.ndarray, float]:
    model, processor = model_hub.get_sam_lora()
    image = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    inputs = processor(images=image, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model(pixel_values=inputs["pixel_values"], multimask_output=False)

    pred_masks = processor.post_process_masks(
        outputs.pred_masks,
        inputs["original_sizes"],
        inputs["reshaped_input_sizes"],
    )
    pred = np.squeeze(pred_masks[0][0].detach().cpu().numpy())
    if pred.ndim != 2:
        pred = pred.reshape(pred.shape[-2], pred.shape[-1])

    mask = (pred > threshold).astype(np.uint8) * 255
    active = pred[pred > threshold]
    confidence = float(active.mean()) if active.size else 0.0
    return mask, confidence


def union_bbox(boxes: list[list[int]], shape: tuple[int, int, int], expand_pct: float = 0.15) -> list[int]:
    h, w = shape[:2]
    x1 = min(b[0] for b in boxes)
    y1 = min(b[1] for b in boxes)
    x2 = max(b[2] for b in boxes)
    y2 = max(b[3] for b in boxes)
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    pad_w = int(bw * expand_pct)
    pad_h = int(bh * expand_pct)
    return [
        max(0, x1 - pad_w),
        max(0, y1 - pad_h),
        min(w, x2 + pad_w),
        min(h, y2 + pad_h),
    ]


def overlay_mask(frame_bgr: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, int, float]:
    overlay = frame_bgr.copy()
    color = overlay.copy()
    color[mask > 0] = (0, 0, 255)
    overlay = cv2.addWeighted(overlay, 0.65, color, 0.35, 0)

    contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)
    area = float(sum(cv2.contourArea(c) for c in contours)) if contours else 0.0
    return overlay, len(contours), area


def draw_detections(frame_bgr: np.ndarray, detections: list[dict[str, Any]]) -> np.ndarray:
    out = frame_bgr.copy()
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        label = str(det["type"])
        conf = float(det.get("confidence", 0.0))
        text = f"{label} {conf:.2f}"
        cv2.rectangle(out, (x1, y1), (x2, y2), (255, 255, 0), 2)
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        ty = y1 - th - 6 if y1 - th - 6 > 0 else y1 + th + 6
        cv2.rectangle(out, (x1, ty), (x1 + tw + 6, ty + th + 4), (255, 255, 0), -1)
        cv2.putText(out, text, (x1 + 3, ty + th), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)
    return out


def process_frame(frame: np.ndarray, frame_idx: int, cfg: RuntimeConfig) -> dict[str, Any]:
    processed = frame
    if cfg.use_preprocess and cfg.mode in {MODE_YOLO26, MODE_YOLO26_MOBILE_SAM}:
        processed = preprocess_frame(frame)

    result = {
        "frame": frame.copy(),
        "defect": False,
        "detections": [],
        "contours": 0,
        "sam_confidence": 0.0,
        "defect_types": [],
        "mask": None,
    }

    if cfg.mode == MODE_SAM:
        mask, conf = run_sam_lora(frame, cfg.sam_threshold)
        overlay, contours, _ = overlay_mask(frame, mask)
        result.update(
            {
                "frame": overlay,
                "defect": bool(np.any(mask)),
                "contours": contours,
                "sam_confidence": conf,
                "mask": mask,
            }
        )
        return result

    if cfg.mode == MODE_MOBILE_SAM:
        h, w = frame.shape[:2]
        mask, conf = run_mobilesam_on_roi(frame, [0, 0, w, h], cfg.sam_threshold)
        overlay, contours, _ = overlay_mask(frame, mask)
        result.update(
            {
                "frame": overlay,
                "defect": bool(np.any(mask)),
                "contours": contours,
                "sam_confidence": conf,
                "mask": mask,
            }
        )
        return result

    detections = run_yolo(processed, cfg.yolo_conf)
    result["detections"] = detections
    result["defect"] = bool(detections)
    result["defect_types"] = sorted({d["type"] for d in detections})

    if cfg.mode == MODE_YOLO26:
        result["frame"] = draw_detections(frame, detections)
        return result

    display = draw_detections(frame, detections)
    if detections:
        union = union_bbox([d["bbox"] for d in detections], frame.shape)
        sam_input = processed if cfg.use_preprocess else frame
        mask, sam_conf = run_mobilesam_on_roi(sam_input, union, cfg.sam_threshold)
        display, contours, area = overlay_mask(display, mask)
        logger.log_detection(
            frame_idx=frame_idx,
            frame=frame,
            overlay=display,
            mask=mask,
            sam_confidence=sam_conf,
            defect_area_px=area,
            detections=detections,
        )
        result["contours"] = contours
        result["sam_confidence"] = sam_conf
        result["mask"] = mask

    result["frame"] = display
    return result


app = Flask(__name__)
CORS(app)

config_store = ConfigStore()
model_hub = ModelHub()
logger = CombinedLogger(LOG_DIR)

runtime_lock = threading.Lock()
latest_snapshot: dict[str, Any] = {
    "image": None,
    "defect": False,
    "detections": [],
    "defect_types": [],
    "contours": 0,
    "mode": config_store.get().mode,
    "use_preprocess": config_store.get().use_preprocess,
    "fps": 0.0,
    "fps_by_mode": {},
    "error": None,
    "timestamp": 0,
}
latest_pattern_summary: dict[str, Any] | None = None

stop_event = threading.Event()


def run_auto_pattern_detection() -> dict[str, Any] | None:
    global latest_pattern_summary
    if not logger.csv_path.exists():
        return None

    out_path = logger.pattern_dir / f"auto_pattern_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    summary = analyze_detection_csv(logger.csv_path, out_json_path=out_path)
    with runtime_lock:
        latest_pattern_summary = summary
    return summary


def camera_worker() -> None:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        with runtime_lock:
            latest_snapshot["error"] = "Could not open camera at index 0"
        return

    fps_queues: dict[str, deque[float]] = {mode: deque(maxlen=30) for mode in SUPPORTED_MODES}
    frame_idx = 0
    last_pattern_run = time.monotonic()

    try:
        while not stop_event.is_set():
            started = time.time()
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.05)
                continue

            frame_idx += 1
            cfg = config_store.get()
            error_msg = None

            try:
                result = process_frame(frame, frame_idx, cfg)
            except Exception as exc:  # keep stream alive even if a model fails
                result = {
                    "frame": frame.copy(),
                    "defect": False,
                    "detections": [],
                    "defect_types": [],
                    "contours": 0,
                    "sam_confidence": 0.0,
                }
                error_msg = str(exc)
                traceback.print_exc()
                cv2.putText(result["frame"], f"Error: {error_msg[:90]}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            elapsed = max(1e-6, time.time() - started)
            fps = 1.0 / elapsed
            fps_queues[cfg.mode].append(fps)
            avg_fps = float(np.mean(fps_queues[cfg.mode])) if fps_queues[cfg.mode] else 0.0

            annotated = result["frame"]
            cv2.putText(annotated, f"Mode: {cfg.mode}", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
            cv2.putText(annotated, f"FPS: {avg_fps:.1f}", (10, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
            if cfg.use_preprocess and cfg.mode in {MODE_YOLO26, MODE_YOLO26_MOBILE_SAM}:
                cv2.putText(annotated, "Preprocess: ON", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2)

            image_b64 = _encode_frame_b64(annotated)
            fps_by_mode = {mode: float(np.mean(q)) if q else 0.0 for mode, q in fps_queues.items()}

            with runtime_lock:
                latest_snapshot.update(
                    {
                        "image": image_b64,
                        "defect": bool(result["defect"]),
                        "detections": result["detections"],
                        "defect_types": result["defect_types"],
                        "contours": int(result["contours"]),
                        "sam_confidence": float(result["sam_confidence"]),
                        "mode": cfg.mode,
                        "use_preprocess": cfg.use_preprocess,
                        "fps": avg_fps,
                        "fps_by_mode": fps_by_mode,
                        "error": error_msg,
                        "timestamp": int(time.time()),
                    }
                )

            if cfg.mode == MODE_YOLO26_MOBILE_SAM and (time.monotonic() - last_pattern_run) >= 300:
                try:
                    run_auto_pattern_detection()
                except Exception:
                    traceback.print_exc()
                last_pattern_run = time.monotonic()

            sleep_for = max(0.0, cfg.process_interval - (time.time() - started))
            if sleep_for > 0:
                time.sleep(sleep_for)
    finally:
        cap.release()


def _safe_path(path_value: str | None) -> Path:
    if not path_value:
        return logger.csv_path
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (BASE_DIR / path).resolve()


def _mjpeg_from_latest():
    while not stop_event.is_set():
        with runtime_lock:
            image = latest_snapshot.get("image")
        if not image:
            time.sleep(0.05)
            continue
        try:
            b64 = image.split(",", 1)[1]
            frame_bytes = base64.b64decode(b64)
        except Exception:
            time.sleep(0.05)
            continue
        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
        time.sleep(0.03)


def _model_status_payload() -> dict[str, Any]:
    paths = model_hub.paths
    return {
        "yolo": {
            "active": str(model_hub.yolo_model_path.resolve()) if model_hub.yolo_model_path else None,
            "candidates": [{"path": str(p.resolve()), "exists": p.exists()} for p in paths.yolo_candidates],
        },
        "mobilesam": {
            "checkpoint_active": str(model_hub.mobile_sam_checkpoint_path.resolve()) if model_hub.mobile_sam_checkpoint_path else None,
            "weights_active": str(model_hub.mobile_sam_weights_path.resolve()) if model_hub.mobile_sam_weights_path else None,
            "checkpoint_candidates": [{"path": str(p.resolve()), "exists": p.exists()} for p in paths.mobile_sam_checkpoint_candidates],
            "weights_candidates": [{"path": str(p.resolve()), "exists": p.exists()} for p in paths.mobile_sam_weights_candidates],
        },
        "sam_lora": {
            "active": str(model_hub.sam_lora_path.resolve()) if model_hub.sam_lora_path else None,
            "candidates": [{"path": str(p.resolve()), "exists": p.exists()} for p in paths.sam_lora_candidates],
        },
    }


def _ensure_models_for_mode(mode: str) -> dict[str, Any]:
    try:
        if mode == MODE_SAM:
            model_hub.get_sam_lora()
        elif mode == MODE_MOBILE_SAM:
            model_hub.get_mobile_sam()
        elif mode == MODE_YOLO26:
            model_hub.get_yolo()
        elif mode == MODE_YOLO26_MOBILE_SAM:
            model_hub.get_yolo()
            model_hub.get_mobile_sam()
    except Exception as exc:
        return {"ok": False, "message": str(exc)}
    return {"ok": True, "message": "Model(s) loaded/validated"}


def _runtime_cfg_for_option(option: str | None, use_preprocess: bool | None) -> RuntimeConfig:
    current = config_store.get()
    if not option:
        cfg = RuntimeConfig(**current.__dict__)
    else:
        key = option.strip().lower()
        if key not in SWITCH_OPTIONS:
            raise ValueError(f"Unsupported switch option: {key}")
        mapped = SWITCH_OPTIONS[key]
        cfg = RuntimeConfig(
            mode=mapped["mode"],
            use_preprocess=bool(mapped["use_preprocess"]),
            process_interval=current.process_interval,
            yolo_conf=current.yolo_conf,
            sam_threshold=current.sam_threshold,
        )
    if use_preprocess is not None:
        cfg.use_preprocess = bool(use_preprocess)
    return cfg


@app.route("/methods", methods=["GET"])
def methods() -> Response:
    return jsonify(
        {
            "status": "success",
            "methods": [
                {"id": MODE_SAM, "name": "SAM", "supports_preprocess": False},
                {"id": MODE_MOBILE_SAM, "name": "MobileSAM", "supports_preprocess": False},
                {"id": MODE_YOLO26, "name": "YOLO26", "supports_preprocess": True},
                {
                    "id": MODE_YOLO26_MOBILE_SAM,
                    "name": "YOLO26 + MobileSAM",
                    "supports_preprocess": True,
                    "notes": "Logs + pattern detection enabled in this mode",
                },
            ],
        }
    )


@app.route("/switch/options", methods=["GET"])
def switch_options() -> Response:
    return jsonify(
        {
            "status": "success",
            "options": [
                {"id": "sam", "label": "SAM"},
                {"id": "mobilesam", "label": "mobileSAM"},
                {"id": "yolo26", "label": "yolo26"},
                {"id": "yolo26_mobilesam", "label": "mobileSAM+yolo26"},
                {"id": "preprocess_yolo26_mobilesam", "label": "preprocess+yolo26+mobileSAM"},
            ],
        }
    )


@app.route("/switch", methods=["POST"])
def switch_method() -> Response:
    payload = request.get_json(silent=True) or {}
    option = str(payload.get("option", "")).strip().lower()
    if option not in SWITCH_OPTIONS:
        return jsonify({"status": "error", "message": f"Unsupported switch option: {option}"}), 400

    cfg_payload = SWITCH_OPTIONS[option]
    cfg = config_store.update(cfg_payload)
    validation = _ensure_models_for_mode(cfg.mode)
    code = 200 if validation["ok"] else 500
    return (
        jsonify(
            {
                "status": "success" if validation["ok"] else "error",
                "selected_option": option,
                "config": cfg.__dict__,
                "model_validation": validation,
                "model_status": _model_status_payload(),
            }
        ),
        code,
    )


@app.route("/config", methods=["GET", "POST"])
def config() -> Response:
    if request.method == "GET":
        cfg = config_store.get()
        return jsonify({"status": "success", "config": cfg.__dict__, "model_status": _model_status_payload()})

    payload = request.get_json(silent=True) or {}
    try:
        cfg = config_store.update(payload)
    except ValueError as exc:
        return jsonify({"status": "error", "message": str(exc)}), 400
    validation = _ensure_models_for_mode(cfg.mode)
    code = 200 if validation["ok"] else 500
    return jsonify({"status": "success" if validation["ok"] else "error", "config": cfg.__dict__, "model_validation": validation, "model_status": _model_status_payload()}), code


@app.route("/models/status", methods=["GET"])
def models_status() -> Response:
    return jsonify({"status": "success", "models": _model_status_payload()})


@app.route("/snapshot", methods=["GET"])
def snapshot() -> Response:
    with runtime_lock:
        image = latest_snapshot.get("image")
        payload = dict(latest_snapshot)

    if image is None:
        return jsonify({"status": "error", "message": "No frames yet"}), 503
    payload["status"] = "success"
    return jsonify(payload)


@app.route("/predict", methods=["POST"])
def predict() -> Response:
    if "image" not in request.files:
        return jsonify({"status": "error", "message": "No image file uploaded"}), 400

    option = request.form.get("option")
    preprocess_raw = request.form.get("use_preprocess")
    use_preprocess = None
    if preprocess_raw is not None:
        use_preprocess = preprocess_raw.strip().lower() in {"1", "true", "yes", "on"}

    try:
        cfg = _runtime_cfg_for_option(option, use_preprocess)
    except ValueError as exc:
        return jsonify({"status": "error", "message": str(exc)}), 400

    validation = _ensure_models_for_mode(cfg.mode)
    if not validation["ok"]:
        return jsonify({"status": "error", "message": validation["message"]}), 500

    try:
        file = request.files["image"]
        pil = Image.open(file.stream).convert("RGB")
        frame = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    except Exception as exc:
        return jsonify({"status": "error", "message": f"Invalid image: {exc}"}), 400

    started = time.time()
    try:
        result = process_frame(frame, frame_idx=0, cfg=cfg)
    except Exception as exc:
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(exc)}), 500
    elapsed = max(1e-6, time.time() - started)

    overlay_b64 = _encode_frame_b64(result["frame"])
    payload: dict[str, Any] = {
        "status": "success",
        "mode": cfg.mode,
        "use_preprocess": cfg.use_preprocess,
        "defect": bool(result["defect"]),
        "defect_types": result["defect_types"],
        "detections": result["detections"],
        "contours": int(result["contours"]),
        "sam_confidence": float(result["sam_confidence"]),
        "fps": float(1.0 / elapsed),
        "overlay_image": overlay_b64,
    }

    if result["mask"] is not None:
        mask = result["mask"].astype(np.uint8)
        mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        payload["binary_mask"] = _encode_frame_b64(mask_rgb)

    return jsonify(payload)


@app.route("/pattern/latest", methods=["GET"])
def pattern_latest() -> Response:
    with runtime_lock:
        summary = latest_pattern_summary
    if summary is None:
        return jsonify({"status": "error", "message": "No pattern summary generated yet"}), 404
    return jsonify({"status": "success", "summary": summary})


@app.route("/pattern/manual", methods=["POST"])
def pattern_manual() -> Response:
    payload = request.get_json(silent=True) or {}
    csv_path = _safe_path(payload.get("csv_path"))
    if not csv_path.exists():
        return jsonify({"status": "error", "message": f"CSV not found: {csv_path}"}), 404

    out_json = payload.get("out_json")
    out_path = _safe_path(out_json) if out_json else logger.pattern_dir / f"manual_pattern_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    try:
        summary = analyze_detection_csv(csv_path, out_json_path=out_path)
    except Exception as exc:
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(exc)}), 500

    with runtime_lock:
        global latest_pattern_summary
        latest_pattern_summary = summary

    return jsonify({"status": "success", "summary": summary})


@app.route("/logs/current", methods=["GET"])
def logs_current() -> Response:
    return jsonify(
        {
            "status": "success",
            "csv_log": str(logger.csv_path.resolve()),
            "json_log": str(logger.json_path.resolve()),
            "pattern_dir": str(logger.pattern_dir.resolve()),
        }
    )


@app.route("/video_feed", methods=["GET"])
def video_feed() -> Response:
    return Response(_mjpeg_from_latest(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/shutdown", methods=["POST"])
def shutdown() -> Response:
    stop_event.set()
    return jsonify({"status": "ok"})


worker = threading.Thread(target=camera_worker, daemon=True)
worker.start()

if __name__ == "__main__":
    try:
        print(f"Backend running on {DEVICE}. Default mode: {config_store.get().mode}")
        app.run(host="0.0.0.0", port=5000, debug=False)
    finally:
        stop_event.set()
