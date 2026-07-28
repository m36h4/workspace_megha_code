#!/usr/bin/env python3
"""
Visualize random COCO ground-truth annotations.

Usage:
    python check_gt.py
"""

import os
import json
import random
import cv2

# ----------------------------
# Configuration
# ----------------------------
ANN_FILE = "dataset/ball_coco/annotations/instances_train.json"
IMG_DIR = "dataset/ball_coco/train"
OUT_DIR = "gt_check"

NUM_IMAGES = 40
RANDOM_SEED = 42

with open(ANN_FILE, "r") as f:
    coco = json.load(f)

images = coco["images"]
annotations = coco["annotations"]
categories = coco["categories"]

cat_map = {c["id"]: c["name"] for c in categories}

ann_map = {}
for ann in annotations:
    ann_map.setdefault(ann["image_id"], []).append(ann)

os.makedirs(OUT_DIR, exist_ok=True)

random.seed(RANDOM_SEED)
selected_images = random.sample(images, min(NUM_IMAGES, len(images)))

print("=" * 60)
print(f"Generating {len(selected_images)} annotated images...")
print("=" * 60)

saved = 0

for img_info in selected_images:
    img_path = os.path.join(IMG_DIR, img_info["file_name"])
    image = cv2.imread(img_path)

    if image is None:
        print(f"[WARNING] Cannot read: {img_path}")
        continue

    anns = ann_map.get(img_info["id"], [])

    for ann in anns:
        x, y, w, h = map(int, ann["bbox"])
        class_name = cat_map.get(ann["category_id"], str(ann["category_id"]))

        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(
            image,
            class_name,
            (x, max(20, y - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
        )

    out_path = os.path.join(OUT_DIR, img_info["file_name"])
    cv2.imwrite(out_path, image)

    print(f"[{saved+1:02d}/{len(selected_images)}] "
          f"{img_info['file_name']} -> {len(anns)} objects")

    saved += 1

print("\nDone!")
print(f"Saved {saved} annotated images to: {OUT_DIR}")
