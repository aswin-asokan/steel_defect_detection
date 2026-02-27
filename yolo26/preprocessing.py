import cv2
import numpy as np
import os

def preprocess_image(img_path, output_dir, patch_size=256, stride=128):
    # Read image
    img = cv2.imread(img_path)

    # 1. Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2. Remove lighting variations using CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    norm = clahe.apply(gray)

    # 3. Slight blur to match dataset texture
    norm = cv2.GaussianBlur(norm, (3, 3), 0)

    # 4. Normalize intensity to 0–255
    norm = cv2.normalize(norm, None, 0, 255, cv2.NORM_MINMAX)

    # 5. Extract patches (dataset-like)
    h, w = norm.shape
    count = 0

    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            patch = norm[y:y+patch_size, x:x+patch_size]

            # Optional: skip very uniform patches
            if np.std(patch) < 5:
                continue

            filename = os.path.join(output_dir, f"patch_{count}.png")
            cv2.imwrite(filename, patch)
            count += 1

    print(f"Saved {count} patches to {output_dir}")


# Usage
input_image = "hot_gouge_6.jpg"
output_folder = "processed_patches"
os.makedirs(output_folder, exist_ok=True)

preprocess_image(input_image, output_folder)