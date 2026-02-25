#!/usr/bin/env python3
"""
YOLO -> MobileSAM real-time defect pipeline with:
 - ROI optimization
 - Saves raw, overlay, mask
 - Draws YOLO boxes + labels on overlay
 - Session CSV with detection rows
 - JSON session summary
"""
import os
import time
import csv
import json
from datetime import datetime
from collections import deque

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from ultralytics import YOLO
from mobile_sam import sam_model_registry

# ================= CONFIG =================
class Config:
    YOLO_MODEL = "runs/detect/train5/weights/best.pt"
    YOLO_CONF = 0.3

    SAM_MODEL_PATH = "mobilesam_defect_optimized/best_model.pth"
    SAM_CHECKPOINT = "mobile_sam.pt"
    SAM_IMG_SIZE = 1024
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    SAM_THRESHOLD = 0.65

    WIDTH = 640
    HEIGHT = 480
    CAMERA_ID = 0

    USE_ROI = True
    ROI_EXPAND_PCT = 0.15

    LOG_DIR = "logs"
    RAW_DIR = os.path.join(LOG_DIR, "raw")
    OVERLAY_DIR = os.path.join(LOG_DIR, "overlay")
    MASK_DIR = os.path.join(LOG_DIR, "mask")
    SAVE_IMAGES = True

    DETECTION_CONFIDENCE_THRESHOLD = 0.6
    FPS_SMOOTHING = 30

config = Config()

# ensure directories exist
os.makedirs(config.LOG_DIR, exist_ok=True)
os.makedirs(config.RAW_DIR, exist_ok=True)
os.makedirs(config.OVERLAY_DIR, exist_ok=True)
os.makedirs(config.MASK_DIR, exist_ok=True)

# ================= LOGGER (CSV + JSON) =================
class ProductionLogger:
    def __init__(self, log_dir):
        self.log_dir = log_dir
        self.records = []
        self.session_start = datetime.now()
        # CSV path
        csv_fname = f"session_{self.session_start.strftime('%Y%m%d_%H%M%S')}.csv"
        self.csv_path = os.path.join(self.log_dir, csv_fname)
        # open csv and write header
        self.csv_file = open(self.csv_path, mode="w", newline="", encoding="utf-8")
        self.csv_writer = csv.writer(self.csv_file)
        header = [
            "timestamp",
            "frame_index",
            "raw_image",
            "overlay_image",
            "mask_image",
            "sam_confidence",
            "defect_area_px",
            "yolo_labels",
            "yolo_confs",
            "yolo_bboxes"  # semicolon-separated x1,y1,x2,y2
        ]
        self.csv_writer.writerow(header)
        self.csv_file.flush()

    def log_detection(self, frame_idx, timestamp, raw_path, overlay_path, mask_path, sam_conf, area_px, yolo_boxes):
        """
        yolo_boxes: list of [x1,y1,x2,y2,conf,class_id,label_str]
        We'll produce semicolon-separated strings for labels/confs/bboxes.
        """
        # build columns for CSV
        labels = []
        confs = []
        bboxes = []
        for b in yolo_boxes:
            # If provided as 6+ length list, handle gracefully
            if len(b) >= 7:
                x1,y1,x2,y2,conf,cls_id,label = b[:7]
            elif len(b) == 6:
                x1,y1,x2,y2,conf,cls_id = b
                label = str(cls_id)
            else:
                # unexpected format
                x1,y1,x2,y2 = b[0],b[1],b[2],b[3]
                conf = b[4] if len(b) > 4 else 0.0
                label = "unknown"
            labels.append(str(label))
            confs.append(f"{float(conf):.3f}")
            bboxes.append(f"{int(x1)},{int(y1)},{int(x2)},{int(y2)}")

        labels_s = ";".join(labels) if labels else ""
        confs_s = ";".join(confs) if confs else ""
        bboxes_s = ";".join(bboxes) if bboxes else ""

        # CSV row (always saved)
        row = [
            timestamp,
            int(frame_idx),
            raw_path if raw_path and os.path.exists(raw_path) else "",
            overlay_path if overlay_path and os.path.exists(overlay_path) else "",
            mask_path if mask_path and os.path.exists(mask_path) else "",
            float(sam_conf),
            float(area_px),
            labels_s,
            confs_s,
            bboxes_s
        ]
        self.csv_writer.writerow(row)
        self.csv_file.flush()

        # JSON logging: only keep entries passing threshold (but still include paths)
        entry = {
            "timestamp": timestamp,
            "frame": int(frame_idx),
            "raw": raw_path if raw_path and os.path.exists(raw_path) else None,
            "overlay": overlay_path if overlay_path and os.path.exists(overlay_path) else None,
            "mask": mask_path if mask_path and os.path.exists(mask_path) else None,
            "sam_confidence": float(sam_conf),
            "defect_area_px": float(area_px),
            "yolo_labels": labels,
            "yolo_confs": confs,
            "yolo_bboxes": bboxes
        }
        if float(sam_conf) >= config.DETECTION_CONFIDENCE_THRESHOLD:
            self.records.append(entry)

    def save_session(self):
        # close csv
        try:
            self.csv_file.close()
        except Exception:
            pass
        # save json summary
        fname = os.path.join(self.log_dir, f"detections_{self.session_start.strftime('%Y%m%d_%H%M%S')}.json")
        payload = {
            "session_start": self.session_start.isoformat(),
            "session_end": datetime.now().isoformat(),
            "total_detections_logged": len(self.records),
            "detections": self.records
        }
        with open(fname, "w") as f:
            json.dump(payload, f, indent=2)
        return fname, self.csv_path

logger = ProductionLogger(config.LOG_DIR)

# ================= MODELS =================
print("Loading YOLO...")
yolo = YOLO(config.YOLO_MODEL)

# attempt to get names mapping
try:
    # ultralytics v8+: yolo.model.names
    names = yolo.model.names
except Exception:
    names = getattr(yolo, "names", None)

print("Loading MobileSAM (this may be slow)...")
if not os.path.exists(config.SAM_CHECKPOINT):
    raise FileNotFoundError(f"SAM checkpoint not found: {config.SAM_CHECKPOINT}")

sam = sam_model_registry["vit_t"](checkpoint=config.SAM_CHECKPOINT)

# load adapter/checkpoint with weights_only=False (trusting your checkpoint)
checkpoint = torch.load(config.SAM_MODEL_PATH, map_location=config.DEVICE, weights_only=False)
state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
sam.load_state_dict(state_dict)
sam.to(config.DEVICE)
sam.eval()
print("Models ready ✅")

# ================= UTILITIES =================
MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
STD  = np.array([58.395, 57.12, 57.375], dtype=np.float32)

def normalize_image_rgb(img_rgb):
    img = img_rgb.astype(np.float32)
    img = (img - MEAN) / STD
    return img

def expand_bbox(x1, y1, x2, y2, img_w, img_h, pct):
    w = max(1, x2 - x1)
    h = max(1, y2 - y1)
    pad_w = int(w * pct)
    pad_h = int(h * pct)
    nx1 = max(0, x1 - pad_w)
    ny1 = max(0, y1 - pad_h)
    nx2 = min(img_w - 1, x2 + pad_w)
    ny2 = min(img_h - 1, y2 + pad_h)
    return int(nx1), int(ny1), int(nx2), int(ny2)

def union_boxes(boxes, img_w, img_h, expand_pct=0.0):
    if len(boxes) == 0:
        return None
    x1 = int(min(b[0] for b in boxes))
    y1 = int(min(b[1] for b in boxes))
    x2 = int(max(b[2] for b in boxes))
    y2 = int(max(b[3] for b in boxes))
    if expand_pct > 0:
        return expand_bbox(x1, y1, x2, y2, img_w, img_h, expand_pct)
    return x1, y1, x2, y2

def run_mobilesam_on_roi(full_bgr, bbox, threshold=config.SAM_THRESHOLD):
    x1, y1, x2, y2 = bbox
    roi = full_bgr[y1:y2, x1:x2]
    if roi.size == 0:
        return np.zeros(full_bgr.shape[:2], dtype=np.uint8), 0.0
    roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    roi_resized = cv2.resize(roi_rgb, (config.SAM_IMG_SIZE, config.SAM_IMG_SIZE))
    roi_norm = normalize_image_rgb(roi_resized)
    tensor = torch.from_numpy(roi_norm).permute(2,0,1).unsqueeze(0).float().to(config.DEVICE)
    with torch.no_grad():
        emb = sam.image_encoder(tensor)
        sparse, dense = sam.prompt_encoder(points=None, boxes=None, masks=None)
        low_res, _ = sam.mask_decoder(
            image_embeddings=emb,
            image_pe=sam.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse,
            dense_prompt_embeddings=dense,
            multimask_output=False
        )
        pred = F.interpolate(low_res, size=(config.SAM_IMG_SIZE, config.SAM_IMG_SIZE), mode="bilinear", align_corners=False)
        prob_map = torch.sigmoid(pred)[0,0].cpu().numpy()
        mask_small = (prob_map > threshold).astype(np.uint8) * 255
    roi_h, roi_w = roi.shape[:2]
    mask_roi = cv2.resize(mask_small, (roi_w, roi_h), interpolation=cv2.INTER_NEAREST)
    mask_full = np.zeros(full_bgr.shape[:2], dtype=np.uint8)
    mask_full[y1:y2, x1:x2] = mask_roi
    conf = float(prob_map[prob_map > threshold].mean()) if np.any(prob_map > threshold) else 0.0
    return mask_full, conf

# ================= MAIN LOOP =================
def main():
    cap = cv2.VideoCapture(config.CAMERA_ID)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.HEIGHT)

    fps_queue = deque(maxlen=config.FPS_SMOOTHING)
    frame_idx = 0

    print("Starting camera — press 'q' to quit")

    try:
        while True:
            t0 = time.time()
            ret, frame = cap.read()
            if not ret:
                print("No frame, exiting.")
                break
            frame_idx += 1
            h, w = frame.shape[:2]

            # YOLO detection
            results = yolo.predict(frame, conf=config.YOLO_CONF, imgsz=512, verbose=False)
            res0 = results[0]
            boxes_xyxy = []  # will hold [x1,y1,x2,y2,conf,cls_id,label]
            if hasattr(res0, "boxes") and len(res0.boxes) > 0:
                # ultralytics stores boxes as tensors; extract them safely
                try:
                    xyxy_tensor = res0.boxes.xyxy  # tensor Nx4
                    conf_tensor = res0.boxes.conf   # tensor N
                    cls_tensor = res0.boxes.cls     # tensor N
                    for i in range(len(xyxy_tensor)):
                        xyxy = xyxy_tensor[i].cpu().numpy()
                        conf = float(conf_tensor[i].cpu().numpy()) if conf_tensor is not None else 1.0
                        cls_id = int(cls_tensor[i].cpu().numpy()) if cls_tensor is not None else -1
                        label = names[cls_id] if (names is not None and cls_id in names) else str(cls_id)
                        boxes_xyxy.append([int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3]), conf, cls_id, label])
                except Exception:
                    # fallback: iterate through results.boxes.__iter__
                    for b in res0.boxes:
                        try:
                            xyxy = b.xyxy.cpu().numpy() if hasattr(b, "xyxy") else b.xyxy
                            conf = float(b.conf.cpu().numpy()) if hasattr(b, "conf") else 1.0
                            cls_id = int(b.cls.cpu().numpy()) if hasattr(b, "cls") else -1
                            label = names[cls_id] if (names is not None and cls_id in names) else str(cls_id)
                            boxes_xyxy.append([int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3]), conf, cls_id, label])
                        except Exception:
                            continue

            defect_detected = len(boxes_xyxy) > 0
            # create initial annotated image from YOLO plot (will be replaced by overlay if SAM runs)
            try:
                annotated = res0.plot()
            except Exception:
                annotated = frame.copy()

            if defect_detected:
                ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                raw_path = os.path.join(config.RAW_DIR, f"{ts}.jpg")
                overlay_path = os.path.join(config.OVERLAY_DIR, f"{ts}.jpg")
                mask_path = os.path.join(config.MASK_DIR, f"{ts}.png")

                # 1) Save raw original frame first
                if config.SAVE_IMAGES:
                    cv2.imwrite(raw_path, frame)

                # 2) Run SAM on union ROI (or full image)
                bboxes_only = [[b[0], b[1], b[2], b[3]] for b in boxes_xyxy]
                union = union_boxes(bboxes_only, w, h, expand_pct=config.ROI_EXPAND_PCT if config.USE_ROI else 0.0)
                if config.USE_ROI and union is not None:
                    mask_full, sam_conf = run_mobilesam_on_roi(frame, union, threshold=config.SAM_THRESHOLD)
                else:
                    # fallback: run on full frame
                    mask_full, sam_conf = run_mobilesam_on_roi(frame, (0,0,w,h), threshold=config.SAM_THRESHOLD)

                # 3) Build overlay: mask + contours + YOLO boxes+labels
                contours, _ = cv2.findContours(mask_full.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                total_area = float(np.sum([cv2.contourArea(c) for c in contours])) if contours else 0.0
                overlay = frame.copy()
                # red mask
                overlay[mask_full > 0] = (0, 0, 255)
                # green contours
                if contours:
                    cv2.drawContours(overlay, contours, -1, (0,255,0), 2)
                # draw YOLO boxes + labels on top
                for b in boxes_xyxy:
                    x1, y1, x2, y2 = int(b[0]), int(b[1]), int(b[2]), int(b[3])
                    conf = float(b[4]) if len(b) > 4 else 0.0
                    cls_id = int(b[5]) if len(b) > 5 else -1
                    
                    # Get label safely
                    if len(b) > 6:
                        label = str(b[6])
                    else:
                        if names is not None and cls_id in names:
                            label = names[cls_id]
                        else:
                            label = str(cls_id)

                    # Draw box
                    cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 255, 0), 2)

                    # Label text
                    text = f"{label} {conf:.2f}"
                    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)

                    ty = y1 - th - 6
                    if ty < 0:
                        ty = y1 + th + 6

                    cv2.rectangle(overlay, (x1, ty), (x1 + tw + 6, ty + th + 4), (255, 255, 0), -1)
                    cv2.putText(overlay, text, (x1 + 3, ty + th), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

                # 4) Save overlay and mask (ensure mask is single-channel PNG)
                if config.SAVE_IMAGES:
                    cv2.imwrite(overlay_path, overlay)
                    cv2.imwrite(mask_path, mask_full)

                # 5) Log to CSV & JSON
                # convert boxes to the format expected by logger: [x1,y1,x2,y2,conf,cls_id,label]
                logger.log_detection(
                    frame_idx,
                    ts,
                    raw_path if config.SAVE_IMAGES else "",
                    overlay_path if config.SAVE_IMAGES else "",
                    mask_path if config.SAVE_IMAGES else "",
                    sam_conf,
                    total_area,
                    boxes_xyxy
                )

                annotated = overlay

            # FPS & display
            fps = 1.0 / max(1e-6, (time.time() - t0))
            fps_queue.append(fps)
            avg_fps = float(np.mean(fps_queue))
            cv2.putText(annotated, f"FPS: {avg_fps:.1f}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)

            cv2.imshow("YOLO + MobileSAM (q to quit)", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except KeyboardInterrupt:
        print("Interrupted by user")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        json_path, csv_path = logger.save_session()
        print("Session JSON saved to:", json_path)
        print("Session CSV saved to:", csv_path)

if __name__ == "__main__":
    main()