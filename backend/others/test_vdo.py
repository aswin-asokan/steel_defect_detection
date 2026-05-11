# realtime_camera_defect_detection.py
# Aligned with latest Flask backend logic

import cv2
import time
import torch
import numpy as np
from PIL import Image, ImageDraw
from transformers import SamModel, SamProcessor
from peft import PeftModel
from skimage import measure
import os

# -------------------------------
# Configuration
# -------------------------------
MODEL_DIR = "sam_steel_lora"
BASE_MODEL = "facebook/sam-vit-base"

CAM_INDEX = 0
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

PROCESS_EVERY_N_FRAMES = 2   # Increase for better FPS
THRESHOLD = 0.5
ALPHA = 0.4
SAVE_DIR = "camera_outputs"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Running on device: {device}")

os.makedirs(SAVE_DIR, exist_ok=True)

# -------------------------------
# Load Model (ONCE)
# -------------------------------
print("🔹 Loading SAM + LoRA model...")
processor = SamProcessor.from_pretrained(MODEL_DIR)
base_model = SamModel.from_pretrained(BASE_MODEL)
model = PeftModel.from_pretrained(base_model, MODEL_DIR)
model.to(device)
model.eval()
print("✅ Model loaded successfully")

# -------------------------------
# Open Camera
# -------------------------------
cap = cv2.VideoCapture(CAM_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

if not cap.isOpened():
    raise RuntimeError("❌ Cannot open camera")

frame_id = 0
last_overlay = None
last_mask = None

prev_time = time.time()
fps = 0.0

# -------------------------------
# Camera Loop
# -------------------------------
try:
    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        frame_id += 1

        # FPS calculation
        now = time.time()
        fps = 0.9 * fps + 0.1 * (1 / (now - prev_time))
        prev_time = now

        # Process every N frames
        if frame_id % PROCESS_EVERY_N_FRAMES == 0:
            # Convert frame to PIL RGB
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame_rgb)

            # Preprocess
            inputs = processor(images=image, return_tensors="pt").to(device)

            # Inference
            with torch.no_grad():
                outputs = model(
                    pixel_values=inputs["pixel_values"],
                    multimask_output=False
                )

            # 🔑 POST-PROCESS MASK (same as backend)
            pred_masks = processor.post_process_masks(
                outputs.pred_masks,
                inputs["original_sizes"],
                inputs["reshaped_input_sizes"]
            )

            pred_mask = np.squeeze(pred_masks[0][0].cpu().numpy())

            # Ensure 2D
            if pred_mask.ndim != 2:
                print("⚠️ Invalid mask shape:", pred_mask.shape)
                continue

            # Threshold
            pred_mask = (pred_mask > THRESHOLD).astype(np.uint8)

            # -------------------------------
            # Contour Extraction
            # -------------------------------
            contours = measure.find_contours(pred_mask, 0.5)

            # -------------------------------
            # Draw Contours
            # -------------------------------
            overlay = image.copy()
            draw = ImageDraw.Draw(overlay)

            for contour in contours:
                contour_xy = [(float(x), float(y)) for y, x in contour]
                draw.line(contour_xy, fill="red", width=3)

            overlay_np = np.array(overlay)

            # -------------------------------
            # Transparent Mask Overlay
            # -------------------------------
            mask_rgb = np.zeros_like(overlay_np)
            mask_rgb[..., 0] = (pred_mask * 255).astype(np.uint8)

            combined = (
                overlay_np * (1 - ALPHA) +
                mask_rgb * ALPHA
            ).astype(np.uint8)

            last_overlay = combined
            last_mask = (pred_mask * 255).astype(np.uint8)

        # Reuse last processed frame if skipping
        display_frame = last_overlay if last_overlay is not None else frame_bgr

        # HUD
        cv2.putText(
            display_frame,
            f"FPS: {fps:.1f} | Contours: {len(contours) if 'contours' in locals() else 0}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

        cv2.putText(
            display_frame,
            "Press 'q' to quit | 's' to save",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        # Show window
        cv2.imshow("Steel Defect Detection - Live", display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s") and last_overlay is not None:
            ts = int(time.time())
            cv2.imwrite(f"{SAVE_DIR}/overlay_{ts}.png", last_overlay)
            cv2.imwrite(f"{SAVE_DIR}/mask_{ts}.png", last_mask)
            print("💾 Frame saved")

finally:
    cap.release()
    cv2.destroyAllWindows()
    print("✔️ Camera closed")
