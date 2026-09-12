"""
yolo_prepare_dataset.py
Converts LabelMe JSON bounding box annotations → YOLO format
and splits dataset into train/val folders.

FOLDER STRUCTURE EXPECTED BEFORE RUNNING:
    caries_dataset/
        images/          ← your panoramic X-rays (.jpg)
        labelme_json/    ← LabelMe JSON files (or same as images/ if default save)

FOLDER STRUCTURE AFTER RUNNING:
    caries_dataset/
        images/
            train/       ← 85% of images
            val/         ← 15% of images
        labels/
            train/       ← YOLO .txt files for train images
            val/         ← YOLO .txt files for val images
        data.yaml        ← YOLOv8 config file

HOW TO LABEL IN LABELME FOR YOLO:
    1. labelme
    2. Open Dir → select caries_dataset/images/
    3. For each X-ray:
         • Right click → Create Rectangle  (NOT polygon or linestrip)
         • Draw a box tightly around each dark caries lesion on tooth surface
         • Label: type  "caries"  → OK
         • Draw boxes for ALL suspicious dark spots on teeth
         • Ctrl+S to save
    4. After labeling ALL images run:
         python yolo_prepare_dataset.py

LABELING TIPS:
    • Only label dark spots ON tooth surfaces (not in gaps between teeth)
    • Box should be tight around the dark area
    • If unsure — label it (better to have false positives in training than miss lesions)
    • Each image should have 0–6 boxes typically
    • Minimum recommended: label 80–100 images
"""

import os
import json
import shutil
import random
import cv2
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

JSON_DIR   = 'caries_dataset/images'     # where LabelMe saves .json by default
IMAGE_DIR  = 'caries_dataset/images'     # original X-rays
OUTPUT_DIR = 'caries_dataset'            # root output folder
VAL_SPLIT  = 0.15                        # 15% validation
LABEL_NAME = 'caries'                    # class name used in LabelMe
SEED       = 42


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def labelme_bbox_to_yolo(points, img_w, img_h):
    """
    Convert LabelMe rectangle points → YOLO format.
    LabelMe rectangle: [[x1,y1],[x2,y1],[x2,y2],[x1,y2]]  or  [[x1,y1],[x2,y2]]
    YOLO format: cx cy w h  (all normalised 0–1)
    """
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)

    cx = ((x1 + x2) / 2) / img_w
    cy = ((y1 + y2) / 2) / img_h
    w  = (x2 - x1) / img_w
    h  = (y2 - y1) / img_h

    # Clamp to [0, 1]
    cx = max(0.0, min(1.0, cx))
    cy = max(0.0, min(1.0, cy))
    w  = max(0.0, min(1.0, w))
    h  = max(0.0, min(1.0, h))

    return cx, cy, w, h


def convert_json_to_yolo(json_path, img_w, img_h):
    """
    Parse one LabelMe JSON and return list of YOLO label strings.
    Returns empty list if no caries found.
    """
    with open(json_path) as f:
        data = json.load(f)

    lines = []
    for shape in data.get('shapes', []):
        label      = shape.get('label', '').lower().strip()
        shape_type = shape.get('shape_type', '').lower()

        if label != LABEL_NAME:
            continue

        if shape_type not in ('rectangle', 'polygon'):
            print(f"  [warn] {os.path.basename(json_path)}: "
                  f"shape_type='{shape_type}' – expected rectangle")
            continue

        cx, cy, w, h = labelme_bbox_to_yolo(shape['points'], img_w, img_h)

        if w < 0.001 or h < 0.001:
            print(f"  [warn] Skipping near-zero box in {json_path}")
            continue

        lines.append(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

    return lines


# ─────────────────────────────────────────────────────────────────────────────
# MAIN CONVERSION + SPLIT
# ─────────────────────────────────────────────────────────────────────────────

def prepare_yolo_dataset():
    print("YOLOv8 Dataset Preparation")
    print("=" * 50)

    img_exts = {'.jpg', '.jpeg', '.png', '.bmp'}

    # Create output folders
    for split in ['train', 'val']:
        os.makedirs(os.path.join(OUTPUT_DIR, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_DIR, 'labels', split), exist_ok=True)

    # Collect all images that have matching JSON
    pairs = []
    for fname in sorted(os.listdir(IMAGE_DIR)):
        stem, ext = os.path.splitext(fname)
        if ext.lower() not in img_exts:
            continue
        json_path = os.path.join(JSON_DIR, stem + '.json')
        img_path  = os.path.join(IMAGE_DIR, fname)
        if os.path.exists(json_path):
            pairs.append((img_path, json_path, stem, ext))

    print(f"Found {len(pairs)} labeled images")

    if len(pairs) == 0:
        print("\nERROR: No JSON files found!")
        print(f"  Expected .json files in: {os.path.abspath(JSON_DIR)}")
        print("  Make sure you labeled images with LabelMe and pressed Ctrl+S")
        return

    # Shuffle and split
    random.seed(SEED)
    random.shuffle(pairs)
    n_val   = max(1, int(len(pairs) * VAL_SPLIT))
    n_train = len(pairs) - n_val
    train_pairs = pairs[:n_train]
    val_pairs   = pairs[n_train:]
    print(f"Split → Train: {len(train_pairs)}  Val: {len(val_pairs)}")

    # Process each split
    total_boxes = 0
    no_label    = 0

    for split, split_pairs in [('train', train_pairs), ('val', val_pairs)]:
        for img_path, json_path, stem, ext in split_pairs:
            # Read image to get dimensions
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"  [skip] Cannot read: {img_path}")
                continue
            img_h, img_w = img.shape

            # Convert annotations
            yolo_lines = convert_json_to_yolo(json_path, img_w, img_h)

            if len(yolo_lines) == 0:
                no_label += 1
                # Still copy image but with empty label file
                # (needed so YOLO knows these are negative examples)

            total_boxes += len(yolo_lines)

            # Copy image
            dst_img = os.path.join(OUTPUT_DIR, 'images', split, stem + ext)
            shutil.copy2(img_path, dst_img)

            # Write label file
            dst_lbl = os.path.join(OUTPUT_DIR, 'labels', split, stem + '.txt')
            with open(dst_lbl, 'w') as f:
                f.write('\n'.join(yolo_lines))

    print(f"\nTotal caries boxes : {total_boxes}")
    print(f"Images no label    : {no_label} (used as negatives)")
    print(f"Avg boxes/image    : {total_boxes/max(1,len(pairs)):.1f}")

    # Write data.yaml
    yaml_path = os.path.join(OUTPUT_DIR, 'data.yaml')
    abs_output = os.path.abspath(OUTPUT_DIR)
    yaml_content = f"""# YOLOv8 Caries Detection Dataset
path  : {abs_output}
train : images/train
val   : images/val

nc    : 1
names : ['{LABEL_NAME}']
"""
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)

    print(f"\nSaved: {yaml_path}")
    print("\nNext step → run:  python yolo_train.py")
    print("=" * 50)


if __name__ == '__main__':
    prepare_yolo_dataset()
