# app.py
# Flask backend with persistent camera capture + /predict and /snapshot endpoints
import os
import io
import time
import base64
import threading
import traceback

import cv2
import torch
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify, Response
from flask_cors import CORS
from transformers import SamModel, SamProcessor
from peft import PeftModel
from skimage import measure

# -------------------------------
# Configuration
# -------------------------------
MODEL_DIR = "sam_steel_lora"
BASE_SAM = "facebook/sam-vit-base"
CAM_INDEX = 0                     # camera index for cv2.VideoCapture
THRESHOLD = 0.5
ALPHA = 0.4                       # overlay alpha
PROCESS_INTERVAL = 0.12           # seconds between processing frames (~8 FPS) - tune this
device = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------------
# Globals (shared between thread & Flask)
# -------------------------------
_latest_lock = threading.Lock()
_latest_frame_b64 = None          # "data:image/jpeg;base64,...."
_latest_defect_flag = False
_latest_contours = 0

_stop_flag = False

# -------------------------------
# Setup Flask & CORS
# -------------------------------
app = Flask(__name__)
CORS(app)
print(f"🚀 Backend running on device: {device}")

# -------------------------------
# Load model & processor once
# -------------------------------
print("🔹 Loading SAM + LoRA model (may take a while)...")
processor = SamProcessor.from_pretrained(MODEL_DIR)
base_model = SamModel.from_pretrained(BASE_SAM)
model = PeftModel.from_pretrained(base_model, MODEL_DIR)
model.to(device)
model.eval()
torch.set_grad_enabled(False)
if device == "cuda":
    torch.backends.cudnn.benchmark = True
print("✅ Model loaded")

# -------------------------------
# Helpers
# -------------------------------
def pil_from_bgr(bgr: np.ndarray) -> Image.Image:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)

def encode_jpeg_to_b64_rgb(rgb_np: np.ndarray) -> str:
    # rgb_np expected shape (H, W, 3), uint8
    bgr = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2BGR)
    ret, jpeg = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    if not ret:
        raise RuntimeError("Failed to encode JPEG")
    b64 = base64.b64encode(jpeg.tobytes()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"

# -------------------------------
# Camera processing thread
# -------------------------------
def camera_worker():
    global _latest_frame_b64, _latest_defect_flag, _latest_contours, _stop_flag
    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        print("❌ Camera open failed in camera thread (index {})".format(CAM_INDEX))
        return

    try:
        while not _stop_flag:
            start_time = time.time()
            ret, frame = cap.read()
            if not ret:
                # no frame: sleep and retry
                time.sleep(0.1)
                continue

            # Convert to PIL RGB image
            image = pil_from_bgr(frame)

            # Preprocess --> model
            try:
                inputs = processor(images=image, return_tensors="pt").to(device)

                with torch.no_grad():
                    outputs = model(pixel_values=inputs["pixel_values"], multimask_output=False)

                # Post-process masks (align to original size)
                pred_masks = processor.post_process_masks(
                    outputs.pred_masks,
                    inputs["original_sizes"],
                    inputs["reshaped_input_sizes"]
                )

                pred_mask = np.squeeze(pred_masks[0][0].cpu().numpy())  # 2D float mask
                if pred_mask.ndim != 2:
                    # try to reshape fallback
                    pred_mask = pred_mask.reshape(pred_mask.shape[-2], pred_mask.shape[-1])

                bin_mask = (pred_mask > THRESHOLD).astype(np.uint8)

                # contours
                contours = measure.find_contours(bin_mask, 0.5)
                contours_count = len(contours)

                # overlay: use the original RGB image and mask
                overlay_rgb = np.array(image).astype(np.uint8)  # (H,W,3)
                mask_rgb = np.zeros_like(overlay_rgb)
                mask_rgb[..., 0] = (bin_mask * 255).astype(np.uint8)
                combined = (overlay_rgb * (1 - ALPHA) + mask_rgb * ALPHA).astype(np.uint8)

                # draw contours using OpenCV (BGR expectation)
                # convert to BGR, draw, convert back to RGB
                combined_bgr = cv2.cvtColor(combined, cv2.COLOR_RGB2BGR)
                for contour in contours:
                    pts = np.array([[int(x), int(y)] for y, x in contour], np.int32)
                    if pts.shape[0] >= 3:
                        cv2.polylines(combined_bgr, [pts], isClosed=True, color=(0,0,255), thickness=2)

                combined_rgb = cv2.cvtColor(combined_bgr, cv2.COLOR_BGR2RGB)

                # encode to base64 for JSON-friendly transport
                frame_b64 = encode_jpeg_to_b64_rgb(combined_rgb)

                # update shared latest
                with _latest_lock:
                    _latest_frame_b64 = frame_b64
                    _latest_defect_flag = bool(bin_mask.any()) and (contours_count > 0)
                    _latest_contours = contours_count

            except Exception as e:
                # keep loop running even if model fails on a frame
                print("⚠️ Exception during frame processing:", e)
                traceback.print_exc()

            # throttle to desired processing rate
            elapsed = time.time() - start_time
            sleep_time = max(0.0, PROCESS_INTERVAL - elapsed)
            time.sleep(sleep_time)

    finally:
        cap.release()
        print("Camera worker exiting, camera released")

# Start camera thread
_camera_thread = threading.Thread(target=camera_worker, daemon=True)
_camera_thread.start()

# -------------------------------
# Flask endpoints
# -------------------------------

# Image upload predict (unchanged behaviour)
def image_to_base64(pil_image: Image.Image) -> str:
    buf = io.BytesIO()
    pil_image.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "image" not in request.files:
            return jsonify({"status": "error", "message": "No image file uploaded"}), 400

        file = request.files["image"]
        image = Image.open(file.stream).convert("RGB")

        inputs = processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(pixel_values=inputs["pixel_values"], multimask_output=False)

        pred_masks = processor.post_process_masks(
            outputs.pred_masks,
            inputs["original_sizes"],
            inputs["reshaped_input_sizes"]
        )
        pred_mask = np.squeeze(pred_masks[0][0].cpu().numpy())
        if pred_mask.ndim != 2:
            pred_mask = pred_mask.reshape(pred_mask.shape[-2], pred_mask.shape[-1])
        pred_mask_bin = (pred_mask > THRESHOLD).astype(np.uint8)

        contours = measure.find_contours(pred_mask_bin, 0.5)

        overlay = np.array(image)
        mask_rgb = np.zeros_like(overlay)
        mask_rgb[..., 0] = pred_mask_bin * 255
        combined = (overlay * (1 - ALPHA) + mask_rgb * ALPHA).astype(np.uint8)

        overlay_b64 = encode_jpeg_to_b64_rgb(combined)
        mask_b64 = encode_jpeg_to_b64_rgb(np.stack([pred_mask_bin*255]*3, axis=-1))

        return jsonify({
            "status": "success",
            "contours": len(contours),
            "overlay_image": overlay_b64,
            "binary_mask": mask_b64
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"status":"error","message":str(e)}), 500

# Snapshot: returns JSON with base64 image and defect flag
@app.route("/snapshot", methods=["GET"])
def snapshot():
    with _latest_lock:
        frame_b64 = _latest_frame_b64
        defect = _latest_defect_flag
        contours = _latest_contours

    if frame_b64 is None:
        return jsonify({"status":"error","message":"No frames yet"}), 503

    return jsonify({
        "status": "success",
        "image": frame_b64,
        "defect": bool(defect),
        "contours": int(contours),
        "timestamp": int(time.time())
    })

# Optional: keep the MJPEG feed for browsers (unchanged)
def generate_mjpeg():
    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        yield b''
        return

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # simple BGR -> JPEG streaming (no processing)
            ret2, jpeg = cv2.imencode('.jpg', frame)
            if not ret2:
                continue
            frame_bytes = jpeg.tobytes()
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")
    finally:
        cap.release()

@app.route("/video_feed")
def video_feed():
    return Response(generate_mjpeg(), mimetype="multipart/x-mixed-replace; boundary=frame")

# -------------------------------
# Shutdown helper (optional)
# -------------------------------
@app.route("/shutdown", methods=["POST"])
def shutdown():
    global _stop_flag
    _stop_flag = True
    return jsonify({"status":"ok"})

# -------------------------------
# Run
# -------------------------------
if __name__ == "__main__":
    try:
        app.run(host="0.0.0.0", port=5000, debug=False)
    finally:
        _stop_flag = True
