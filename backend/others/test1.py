# ==========================================================
# FINAL MobileSAM INFERENCE
# OUTPUT: Binary Mask + Defect Overlay
# ==========================================================

import torch
import torch.nn.functional as F
import cv2
import numpy as np
from mobile_sam import sam_model_registry
import matplotlib.pyplot as plt

# ---------------- CONFIG ----------------
IMAGE_PATH = "sample/example3.jpg"
MODEL_PATH = "mobilesam_defect_optimized/best_model.pth"

OUTPUT_MASK_PATH = "result_binary_mask.png"
OUTPUT_OVERLAY_PATH = "result_defect_overlay.png"

IMG_SIZE = 1024
THRESHOLD = 0.5

device = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------- LOAD MODEL ----------------
sam = sam_model_registry["vit_t"](checkpoint="mobile_sam.pt")

checkpoint = torch.load(MODEL_PATH, map_location=device)
if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    sam.load_state_dict(checkpoint["model_state_dict"])
else:
    sam.load_state_dict(checkpoint)

sam.to(device)
sam.eval()

print("✅ Model loaded")

# ---------------- LOAD & PREPROCESS IMAGE ----------------
img_bgr = cv2.imread(IMAGE_PATH)
if img_bgr is None:
    raise FileNotFoundError(f"❌ Image not found: {IMAGE_PATH}")

img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
orig_h, orig_w = img_rgb.shape[:2]

img = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE)).astype(np.float32)

# SAME normalization as training
mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
std  = np.array([58.395, 57.12, 57.375], dtype=np.float32)
img = (img - mean) / std

img_tensor = (
    torch.from_numpy(img)
    .permute(2, 0, 1)
    .unsqueeze(0)
    .float()
    .to(device)
)

# ---------------- INFERENCE ----------------
with torch.no_grad():
    image_embedding = sam.image_encoder(img_tensor)

    sparse, dense = sam.prompt_encoder(
        points=None, boxes=None, masks=None
    )

    low_res_mask, _ = sam.mask_decoder(
        image_embeddings=image_embedding,
        image_pe=sam.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse,
        dense_prompt_embeddings=dense,
        multimask_output=False
    )

    pred = F.interpolate(
        low_res_mask,
        size=(IMG_SIZE, IMG_SIZE),
        mode="bilinear",
        align_corners=False
    )

    mask = (torch.sigmoid(pred)[0, 0] > THRESHOLD).cpu().numpy().astype(np.uint8)

# ---------------- POSTPROCESS ----------------
mask = cv2.resize(mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

# binary mask image (0 or 255)
binary_mask = (mask * 255).astype(np.uint8)

# overlay image
overlay_rgb = img_rgb.copy()
overlay_rgb[mask == 1] = [255, 0, 0]   # red defect

# ---------------- SAVE OUTPUTS ----------------
cv2.imwrite(OUTPUT_MASK_PATH, binary_mask)
cv2.imwrite(
    OUTPUT_OVERLAY_PATH,
    cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR)
)

print(f"✅ Binary mask saved as: {OUTPUT_MASK_PATH}")
print(f"✅ Overlay image saved as: {OUTPUT_OVERLAY_PATH}")

# ---------------- DISPLAY OUTPUTS ----------------
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.title("Binary Defect Mask")
plt.imshow(binary_mask, cmap="gray")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.title("Defect Overlay")
plt.imshow(overlay_rgb)
plt.axis("off")

plt.tight_layout()
plt.show()
