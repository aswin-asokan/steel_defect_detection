#!/usr/bin/env python3
"""
prepare_neudet.py

Converts Pascal VOC XML annotations (NEU-DET) to YOLO format labels,
copies images into neu_det/images/{train,val}, labels into neu_det/labels/{train,val},
and writes neu_det/data.yaml (absolute path).

Usage:
    python prepare_neudet.py

Adjust settings at top of file as needed.
"""

import os
import random
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

# ========== USER SETTINGS ==========
NEU_ROOT = "NEU-DET"             # root where ANNOTATIONS/IMAGES exist
XML_DIR = os.path.join(NEU_ROOT, "ANNOTATIONS")
IMG_DIR = os.path.join(NEU_ROOT, "IMAGES")
OUT_ROOT = "neu_det"             # output dataset folder for ultralytics
TRAIN_SPLIT = 0.8                # fraction for training
RANDOM_SEED = 0
# class order used by NEU-DET
CLASSES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled_in_scale",
    "scratches"
]
# allowed image extensions to search for
IMG_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]
# ===================================

random.seed(RANDOM_SEED)

os.makedirs(OUT_ROOT, exist_ok=True)

# target directories
IMAGES_TRAIN = os.path.join(OUT_ROOT, "images", "train")
IMAGES_VAL   = os.path.join(OUT_ROOT, "images", "val")
LABELS_TRAIN = os.path.join(OUT_ROOT, "labels", "train")
LABELS_VAL   = os.path.join(OUT_ROOT, "labels", "val")

for d in [IMAGES_TRAIN, IMAGES_VAL, LABELS_TRAIN, LABELS_VAL]:
    os.makedirs(d, exist_ok=True)

def convert_box(size, box):
    w, h = size
    xmin, ymin, xmax, ymax = box
    x_center = (xmin + xmax) / 2.0 / w
    y_center = (ymin + ymax) / 2.0 / h
    width = (xmax - xmin) / w
    height = (ymax - ymin) / h
    return x_center, y_center, width, height

def find_image_file_by_stem(stem, img_dir):
    """Look for a file with the given stem and allowed extensions in img_dir."""
    for ext in IMG_EXTS:
        candidate = os.path.join(img_dir, stem + ext)
        if os.path.exists(candidate):
            return candidate
    # also try case-insensitive or nested subfolders
    # quick walk (only first level)
    for root, _, files in os.walk(img_dir):
        for f in files:
            if Path(f).stem.lower() == stem.lower():
                return os.path.join(root, f)
    return None

# collect xmls
xml_files = [f for f in os.listdir(XML_DIR) if f.lower().endswith(".xml")]
xml_files.sort()
print(f"Found {len(xml_files)} XML annotations in {XML_DIR}")

pairs = []   # list of tuples (image_path, label_text_content, stem)

missing_images = []
converted = 0

for xml_name in xml_files:
    xml_path = os.path.join(XML_DIR, xml_name)
    tree = ET.parse(xml_path)
    root = tree.getroot()

    size = root.find("size")
    if size is None:
        print("Warning: <size> missing in", xml_name)
        continue
    w = int(size.find("width").text)
    h = int(size.find("height").text)

    # derive image stem from xml name (strip .xml)
    stem = Path(xml_name).stem

    # find the image file with this stem
    img_file = find_image_file_by_stem(stem, IMG_DIR)
    if img_file is None:
        missing_images.append(stem)
        continue

    # build label lines
    lines = []
    for obj in root.iter("object"):
        cls = obj.find("name").text
        if cls not in CLASSES:
            # skip unknown classes
            continue
        cls_id = CLASSES.index(cls)
        xmlbox = obj.find("bndbox")
        box = (
            float(xmlbox.find("xmin").text),
            float(xmlbox.find("ymin").text),
            float(xmlbox.find("xmax").text),
            float(xmlbox.find("ymax").text),
        )
        x, y, bw, bh = convert_box((w, h), box)
        lines.append(f"{cls_id} {x:.6f} {y:.6f} {bw:.6f} {bh:.6f}")

    pairs.append((img_file, lines, stem))
    converted += 1

print(f"Convertible pairs found: {converted}")
if missing_images:
    print(f"Missing image files for {len(missing_images)} xmls. Example missing: {missing_images[:10]}")
    print("Please ensure IMAGES folder contains the image files with matching basenames.")
    # continue anyway so user can fix missing ones

# split into train/val
random.shuffle(pairs)
n_total = len(pairs)
n_train = int(n_total * TRAIN_SPLIT)
train_pairs = pairs[:n_train]
val_pairs = pairs[n_train:]

print(f"Split: {n_train} train / {n_total-n_train} val")

# Copy files & write label files
def copy_and_write(pairs_list, out_images_dir, out_labels_dir):
    copied = 0
    missing = 0
    for img_path, label_lines, stem in pairs_list:
        # image dest
        dest_img = os.path.join(out_images_dir, os.path.basename(img_path))
        try:
            shutil.copy(img_path, dest_img)
            copied += 1
        except Exception as e:
            print("Error copying", img_path, "->", dest_img, e)
            missing += 1
            continue

        # label dest
        label_dest = os.path.join(out_labels_dir, stem + ".txt")
        with open(label_dest, "w") as f:
            for ln in label_lines:
                f.write(ln + "\n")
    return copied, missing

t_copied, t_missing = copy_and_write(train_pairs, IMAGES_TRAIN, LABELS_TRAIN)
v_copied, v_missing = copy_and_write(val_pairs, IMAGES_VAL, LABELS_VAL)

print(f"Copied: train images {t_copied}, val images {v_copied}")
if t_missing + v_missing > 0:
    print("Some images could not be copied:", t_missing + v_missing)

# write data.yaml with absolute path
abs_root = os.path.abspath(OUT_ROOT)
data_yaml = os.path.join(OUT_ROOT, "data.yaml")
with open(data_yaml, "w") as f:
    f.write(f"path: {abs_root}\n")
    f.write("train: images/train\n")
    f.write("val: images/val\n")
    f.write("names:\n")
    for i, name in enumerate(CLASSES):
        f.write(f"  {i}: {name}\n")

print("Wrote", data_yaml)
print("Done. You can now train with:")
print(f"  from ultralytics import YOLO\n  model = YOLO('yolo26n.pt')\n  model.train(data='{data_yaml}', epochs=60, imgsz=512, batch=2)")

# final diagnostics
print("\nDiagnostics:")
print(" - annotation xml count:", len(xml_files))
print(" - convertible pairs:", converted)
print(" - train images copied:", t_copied)
print(" - val images copied:", v_copied)
print(" - missing image basenames (sample):", missing_images[:20])
