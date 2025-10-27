# ==========================================================
# Project: Steel Defect Detection - Inference Script (Aligned Overlay)
# Author: Bineesha KP
# ==========================================================

import torch
import numpy as np
from PIL import Image, ImageDraw
from transformers import SamModel, SamProcessor
from peft import PeftModel
from skimage import measure
import matplotlib.pyplot as plt
import os

# -------------------------------
# Configuration
# -------------------------------
MODEL_DIR = "sam_steel_lora"
IMAGE_PATH = "sample/example1.jpeg"
OUTPUT_PATH = "sample/example_result.jpeg"

device = "cuda" if torch.cuda.is_available() else "cpu"
print("🚀 Running inference on:", device)

# -------------------------------
# Load Model & Processor
# -------------------------------
print("🔹 Loading trained SAM + LoRA model...")
processor = SamProcessor.from_pretrained(MODEL_DIR)
base_model = SamModel.from_pretrained("facebook/sam-vit-base")
model = PeftModel.from_pretrained(base_model, MODEL_DIR)
model.to(device)
model.eval()

# -------------------------------
# Load Image
# -------------------------------
image = Image.open(IMAGE_PATH).convert("RGB")
orig_w, orig_h = image.size
inputs = processor(images=image, return_tensors="pt").to(device)

# -------------------------------
# Predict Defect Mask
# -------------------------------
print("🔹 Predicting defect mask...")
with torch.no_grad():
    outputs = model(pixel_values=inputs["pixel_values"], multimask_output=False)
    pred_mask = outputs.pred_masks.squeeze().cpu().numpy()

# -------------------------------
# Handle Multi-channel Output
# -------------------------------
if pred_mask.ndim == 3:
    print(f"⚙️ Detected {pred_mask.shape[0]} mask channels — combining them...")
    pred_mask = np.mean(pred_mask, axis=0)

# Ensure 2D
if pred_mask.ndim != 2:
    raise ValueError(f"❌ Final mask must be 2D but got {pred_mask.shape}")

# Threshold mask
pred_mask = (pred_mask > 0.5).astype(np.uint8)
# -------------------------------
# Resize mask back to original image size
# -------------------------------
mask_img = Image.fromarray(pred_mask * 255).resize((orig_w, orig_h), resample=Image.NEAREST)
pred_mask_resized = np.array(mask_img).astype(np.uint8)  # shape: (orig_h, orig_w)
pred_mask_resized_norm = pred_mask_resized / 255.0  # normalize 0-1 for blending

# -------------------------------
# Contour Extraction
# -------------------------------
contours = measure.find_contours(pred_mask_resized_norm, 0.5)
print(f"✅ Found {len(contours)} defect contour(s).")

# -------------------------------
# Draw Contours on Original Image
# -------------------------------
overlay = image.copy()
draw = ImageDraw.Draw(overlay)

for contour in contours:
    # Swap (row, col) -> (x, y) for PIL/ImageDraw
    contour_xy = [(float(x), float(y)) for y, x in contour]
    draw.line(contour_xy, fill="red", width=3)

# -------------------------------
# Transparent Overlay for Better Visualization
# -------------------------------
overlay_array = np.array(overlay)  # (orig_h, orig_w, 3)
mask_rgb = np.zeros_like(overlay_array)
mask_rgb[..., 0] = (pred_mask_resized_norm * 255).astype(np.uint8)  # red channel

alpha = 0.4  # transparency
combined = (overlay_array * (1 - alpha) + mask_rgb * alpha).astype(np.uint8)
final_img = Image.fromarray(combined)

# -------------------------------
# Save and Display
# -------------------------------
final_img.save(OUTPUT_PATH)
print(f"✅ Perfectly aligned defect overlay saved to: {OUTPUT_PATH}")

plt.figure(figsize=(10, 8))
plt.imshow(final_img)
plt.axis("off")
plt.show()

# -------------------------------
# Save Binary Mask
# -------------------------------
mask_path = os.path.join(os.path.dirname(OUTPUT_PATH), "example_mask.jpeg")
mask_img.save(mask_path)
print(f"💾 Binary mask saved to: {mask_path}")
