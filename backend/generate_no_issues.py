import json
import random
from datetime import datetime, timedelta

# ================= CONFIG =================
OUTPUT_FILE = "detections_normal_session.json"
TOTAL_FRAMES = 300

DEFECT_TYPES = ["scratches", "crazing", "patches", "inclusion"]
PROB_DEFECT = 0.25      # only 25% frames have defects
MAX_DEFECTS_PER_FRAME = 2
# ==========================================


def random_bbox():
    """Generate random bbox across image (simulate random locations)"""
    x1 = random.randint(0, 500)
    y1 = random.randint(0, 400)
    w = random.randint(30, 150)
    h = random.randint(20, 120)
    x2 = min(640, x1 + w)
    y2 = min(480, y1 + h)
    return f"{x1},{y1},{x2},{y2}"


def random_timestamp(start_time, frame_idx):
    """Simulate timestamps like your format"""
    t = start_time + timedelta(milliseconds=100 * frame_idx)
    return t.strftime("%Y%m%d_%H%M%S_%f")


# ================= GENERATE DATA =================
start_time = datetime.now()
detections = []

for frame in range(TOTAL_FRAMES):

    if random.random() > PROB_DEFECT:
        # No defect in this frame
        continue

    num_defects = random.randint(1, MAX_DEFECTS_PER_FRAME)

    labels = []
    confs = []
    bboxes = []

    for _ in range(num_defects):
        label = random.choice(DEFECT_TYPES)
        labels.append(label)
        confs.append(str(round(random.uniform(0.3, 0.7), 3)))
        bboxes.append(random_bbox())

    detections.append({
        "timestamp": random_timestamp(start_time, frame),
        "frame": frame,
        "raw": f"logs/raw/frame_{frame}.jpg",
        "overlay": f"logs/overlay/frame_{frame}.jpg",
        "mask": f"logs/mask/frame_{frame}.png",
        "sam_confidence": round(random.uniform(0.85, 0.98), 3),
        "defect_area_px": round(random.uniform(500, 15000), 1),
        "yolo_labels": labels,
        "yolo_confs": confs,
        "yolo_bboxes": bboxes
    })


session_data = {
    "session_start": start_time.isoformat(),
    "session_end": (start_time + timedelta(seconds=TOTAL_FRAMES*0.1)).isoformat(),
    "total_detections_logged": len(detections),
    "detections": detections
}

with open(OUTPUT_FILE, "w") as f:
    json.dump(session_data, f, indent=2)

print(f"Generated normal session JSON: {OUTPUT_FILE}")
print(f"Total detection entries: {len(detections)}")