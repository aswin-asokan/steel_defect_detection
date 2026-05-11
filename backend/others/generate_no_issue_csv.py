import os
import random
import pandas as pd
import numpy as np
from PIL import Image
from datetime import datetime, timedelta

# ================= CONFIG =================
OUTPUT_CSV = "logs/no_issue_session.csv"
IMAGE_DIR = "logs/no_issue_images"
TOTAL_SAMPLES = 60

IMG_SIZE = (640, 480)
DEFECT_TYPES = ["scratches", "crazing", "patches", "inclusion"]

# Ensure directories
os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs("logs", exist_ok=True)

# ==========================================

def random_bbox():
    x1 = random.randint(0, 500)
    y1 = random.randint(0, 350)
    w = random.randint(20, 120)
    h = random.randint(20, 120)
    x2 = min(639, x1 + w)
    y2 = min(479, y1 + h)
    return f"{x1},{y1},{x2},{y2}"

def generate_random_image(path):
    """Generate visually different noise images"""
    img = np.random.randint(0, 255, (IMG_SIZE[1], IMG_SIZE[0], 3), dtype=np.uint8)
    Image.fromarray(img).save(path)

rows = []
start_time = datetime.now()

for i in range(TOTAL_SAMPLES):

    # Spread timestamps → avoid time clustering
    ts = start_time + timedelta(minutes=i*2)

    # Random image
    img_name = f"img_{i}.jpg"
    img_path = os.path.join(IMAGE_DIR, img_name)
    generate_random_image(img_path)

    # Random defect type (mixed)
    label = random.choice(DEFECT_TYPES)

    # Random bbox (different locations)
    bbox = random_bbox()

    rows.append({
        "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
        "frame_index": i,
        "raw_image": img_path,
        "overlay_image": img_path,
        "mask_image": img_path,
        "sam_confidence": round(random.uniform(0.85, 0.98), 3),
        "defect_area_px": random.randint(500, 15000),
        "yolo_labels": label,
        "yolo_confs": round(random.uniform(0.3, 0.7), 3),
        "yolo_bboxes": bbox
    })

df = pd.DataFrame(rows)
df.to_csv(OUTPUT_CSV, index=False)

print("Generated NO-ISSUE CSV:")
print(OUTPUT_CSV)
print("Total samples:", len(df))