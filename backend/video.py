#!/usr/bin/env python3

# ==========================================================
# YOLO + MobileSAM VIDEO DEFECT DETECTION
# INPUT  : VIDEO (.mp4 / .webm)
# OUTPUT : Annotated Video
#
# FEATURES:
# - YOLO defect detection
# - MobileSAM segmentation
# - Defect boundary contours
# - Defect labels
# - Semi-transparent overlay
# ==========================================================

import cv2
import torch
import numpy as np
import torch.nn.functional as F

from ultralytics import YOLO
from mobile_sam import sam_model_registry

# ==========================================================
# CONFIG
# ==========================================================

VIDEO_INPUT = "output.webm"
VIDEO_OUTPUT = "output_annotated.mp4"

YOLO_MODEL = "yolo26/runs/detect/train5/weights/best.pt"
YOLO_CONF = 0.30

SAM_MODEL_PATH = "mobilesam_defect_optimized/best_model.pth"
SAM_CHECKPOINT = "mobile_sam.pt"

IMG_SIZE = 1024
THRESHOLD = 0.5

# FORCE CPU
# (change to "cuda" later after fixing CUDA setup)
DEVICE = "cpu"

# ==========================================================
# LOAD YOLO
# ==========================================================

print("Loading YOLO...")

yolo = YOLO(YOLO_MODEL)

try:
    CLASS_NAMES = yolo.model.names
except Exception:
    CLASS_NAMES = yolo.names

# ==========================================================
# LOAD MobileSAM
# ==========================================================

print("Loading MobileSAM...")

sam = sam_model_registry["vit_t"](
    checkpoint=SAM_CHECKPOINT
)

# IMPORTANT:
# PyTorch 2.6 requires weights_only=False
checkpoint = torch.load(
    SAM_MODEL_PATH,
    map_location=DEVICE,
    weights_only=False
)

if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    sam.load_state_dict(checkpoint["model_state_dict"])
else:
    sam.load_state_dict(checkpoint)

sam.to(DEVICE)
sam.eval()

print("Models Loaded Successfully!")

# ==========================================================
# NORMALIZATION
# ==========================================================

MEAN = np.array(
    [123.675, 116.28, 103.53],
    dtype=np.float32
)

STD = np.array(
    [58.395, 57.12, 57.375],
    dtype=np.float32
)

# ==========================================================
# SEGMENTATION FUNCTION
# ==========================================================

def segment_defect(frame_bgr, bbox):

    x1, y1, x2, y2 = bbox

    H, W = frame_bgr.shape[:2]

    # ======================================================
    # ROI EXPANSION
    # ======================================================

    pad_x = int((x2 - x1) * 0.25)
    pad_y = int((y2 - y1) * 0.25)

    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)

    x2 = min(W, x2 + pad_x)
    y2 = min(H, y2 + pad_y)

    roi_bgr = frame_bgr[y1:y2, x1:x2]

    if roi_bgr.size == 0:
        return np.zeros((H, W), dtype=np.uint8)

    # ======================================================
    # PREPROCESS
    # ======================================================

    roi_rgb = cv2.cvtColor(
        roi_bgr,
        cv2.COLOR_BGR2RGB
    )

    roi_h, roi_w = roi_rgb.shape[:2]

    img = cv2.resize(
        roi_rgb,
        (IMG_SIZE, IMG_SIZE)
    ).astype(np.float32)

    img = (img - MEAN) / STD

    img_tensor = (
        torch.from_numpy(img)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .float()
        .to(DEVICE)
    )

    # ======================================================
    # MobileSAM INFERENCE
    # ======================================================

    with torch.no_grad():

        image_embedding = sam.image_encoder(img_tensor)

        sparse_embeddings, dense_embeddings = sam.prompt_encoder(
            points=None,
            boxes=None,
            masks=None
        )

        low_res_masks, _ = sam.mask_decoder(
            image_embeddings=image_embedding,
            image_pe=sam.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False
        )

        pred = F.interpolate(
            low_res_masks,
            size=(IMG_SIZE, IMG_SIZE),
            mode="bilinear",
            align_corners=False
        )

        prob = torch.sigmoid(pred)[0, 0]

        mask = (
            prob > THRESHOLD
        ).cpu().numpy().astype(np.uint8)

    # ======================================================
    # RESTORE ORIGINAL ROI SIZE
    # ======================================================

    mask = cv2.resize(
        mask,
        (roi_w, roi_h),
        interpolation=cv2.INTER_NEAREST
    )

    # ======================================================
    # MORPHOLOGICAL CLEANUP
    # ======================================================

    kernel = np.ones((5, 5), np.uint8)

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    # ======================================================
    # INSERT INTO FULL FRAME
    # ======================================================

    full_mask = np.zeros((H, W), dtype=np.uint8)

    full_mask[y1:y2, x1:x2] = mask

    return full_mask

# ==========================================================
# VIDEO SETUP
# ==========================================================

cap = cv2.VideoCapture(VIDEO_INPUT)

if not cap.isOpened():
    raise Exception(f"Cannot open video: {VIDEO_INPUT}")

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

print(f"Video Size : {width}x{height}")
print(f"FPS        : {fps}")

# MP4 OUTPUT
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

out = cv2.VideoWriter(
    VIDEO_OUTPUT,
    fourcc,
    fps,
    (width, height)
)

# ==========================================================
# PROCESS VIDEO
# ==========================================================

frame_count = 0

print("Processing video...")

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_count += 1

    annotated = frame.copy()

    # ======================================================
    # YOLO DETECTION
    # ======================================================

    results = yolo.predict(
        frame,
        conf=YOLO_CONF,
        imgsz=640,
        verbose=False
    )

    result = results[0]

    # ======================================================
    # IF DETECTIONS FOUND
    # ======================================================

    if hasattr(result, "boxes") and len(result.boxes) > 0:

        boxes = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy()

        for box, conf, cls_id in zip(
            boxes,
            confs,
            classes
        ):

            x1, y1, x2, y2 = map(int, box)

            label = CLASS_NAMES[int(cls_id)]

            # ==================================================
            # SEGMENT DEFECT
            # ==================================================

            mask = segment_defect(
                frame,
                (x1, y1, x2, y2)
            )

            # ==================================================
            # RED TRANSPARENT OVERLAY
            # ==================================================

            red_layer = np.zeros_like(frame)
            red_layer[:] = (0, 0, 255)

            alpha = 0.35

            blended = cv2.addWeighted(
                annotated,
                1 - alpha,
                red_layer,
                alpha,
                0
            )

            annotated = np.where(
                mask[:, :, None] == 1,
                blended,
                annotated
            )

            # ==================================================
            # CONTOURS
            # ==================================================

            contours, _ = cv2.findContours(
                (mask * 255).astype(np.uint8),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )

            cv2.drawContours(
                annotated,
                contours,
                -1,
                (0, 255, 0),
                2
            )

            # ==================================================
            # BOUNDING BOX
            # ==================================================

            cv2.rectangle(
                annotated,
                (x1, y1),
                (x2, y2),
                (255, 255, 0),
                2
            )

            # ==================================================
            # LABEL
            # ==================================================

            text = f"{label} {conf:.2f}"

            (tw, th), _ = cv2.getTextSize(
                text,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                2
            )

            label_y = y1 - 10

            if label_y < 20:
                label_y = y1 + 30

            cv2.rectangle(
                annotated,
                (x1, label_y - th - 10),
                (x1 + tw + 10, label_y + 5),
                (255, 255, 0),
                -1
            )

            cv2.putText(
                annotated,
                text,
                (x1 + 5, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 0),
                2
            )

    # ======================================================
    # FPS DISPLAY
    # ======================================================

    cv2.putText(
        annotated,
        f"Frame: {frame_count}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    # ======================================================
    # SAVE FRAME
    # ======================================================

    out.write(annotated)

    # ======================================================
    # DISPLAY
    # ======================================================

    cv2.imshow(
        "Steel Defect Detection",
        annotated
    )

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    print(f"Processed Frame: {frame_count}", end="\r")

# ==========================================================
# CLEANUP
# ==========================================================

cap.release()
out.release()

cv2.destroyAllWindows()

print("\nProcessing Complete!")
print(f"Saved Output Video: {VIDEO_OUTPUT}")