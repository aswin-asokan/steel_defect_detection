import os
import io
import base64
import torch
import numpy as np
from PIL import Image, ImageDraw
from transformers import SamModel, SamProcessor
from peft import PeftModel
from skimage import measure
from flask import Flask, request, jsonify
from flask_cors import CORS
MODEL_DIR = "sam_steel_lora"
device = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize Flask App
app = Flask(__name__)
CORS(app)
print(f" Backend running on: {device}")

# Load Model Once
print(" Loading SAM + LoRA model...")
processor = SamProcessor.from_pretrained(MODEL_DIR)
base_model = SamModel.from_pretrained("facebook/sam-vit-base")
model = PeftModel.from_pretrained(base_model, MODEL_DIR)
model.to(device)
model.eval()
print(" Model loaded successfully!")

# Helper: Convert Image to Base64
def image_to_base64(image: Image.Image):
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

# Route: Prediction Endpoint
@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"status": "error", "message": "No image file uploaded"}), 400

    # Read uploaded image
    file = request.files["image"]
    image = Image.open(file.stream).convert("RGB")
    orig_w, orig_h = image.size

    # Preprocess
    inputs = processor(images=image, return_tensors="pt").to(device)

    # Predict defect mask with alignment
    print(" Predicting defect mask...")
    with torch.no_grad():
        outputs = model(pixel_values=inputs["pixel_values"], multimask_output=False)

    # Align predicted mask to original image size
    pred_masks = processor.post_process_masks(
        outputs.pred_masks,
        inputs["original_sizes"],
        inputs["reshaped_input_sizes"]
    )
    pred_mask = np.squeeze(pred_masks[0][0].cpu().numpy())

    # Ensure mask is 2D
    if pred_mask.ndim != 2:
        return jsonify({"status": "error", "message": f"Expected 2D mask but got shape: {pred_mask.shape}"}), 500

    # Threshold mask
    pred_mask = (pred_mask > 0.5).astype(np.uint8)
    mask_img = Image.fromarray((pred_mask * 255).astype(np.uint8))

    # Extract contours
    print(" Extracting contours...")
    pred_mask_norm = pred_mask / 1.0
    contours = measure.find_contours(pred_mask_norm, 0.5)

    # Draw contours on original image
    overlay = image.copy()
    draw = ImageDraw.Draw(overlay)
    for contour in contours:
        contour_xy = [(float(x), float(y)) for y, x in contour]
        draw.line(contour_xy, fill="red", width=3)

    # Transparent overlay blend
    overlay_array = np.array(overlay)
    mask_rgb = np.zeros_like(overlay_array)
    mask_rgb[..., 0] = (pred_mask * 255).astype(np.uint8)
    alpha = 0.4
    combined = (overlay_array * (1 - alpha) + mask_rgb * alpha).astype(np.uint8)
    final_img = Image.fromarray(combined)

    # Encode images to base64
    overlay_b64 = image_to_base64(final_img)
    mask_b64 = image_to_base64(mask_img)

    # Response
    return jsonify({
        "status": "success",
        "contours": len(contours),
        "overlay_image": f"data:image/jpeg;base64,{overlay_b64}",
        "binary_mask": f"data:image/jpeg;base64,{mask_b64}"
    })


# Main Entry
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
