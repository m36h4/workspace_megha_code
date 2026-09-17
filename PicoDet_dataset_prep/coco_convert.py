#!/usr/bin/env python3
"""
Generic COCO Conversion Script
===============================

Converts a flat dataset (images in one folder, matching JSON
annotations in another folder — no nested subfolders) into COCO
format, with an optional random train/val/test split.

Expected input layout:

    images_folder/
        img1.jpg
        img2.png
        ...

    annotations_folder/
        img1.json
        img2.json
        ...

Each JSON annotation is expected to look like:

    {
        "dimensions": [height, width],   # optional, falls back to
                                          # reading the actual image
        "data": {
            "ball": [
                {"entire": {"rect": [x1, y1, x2, y2]}},
                ...
            ]
        }
    }

Usage:

    python coco_convert.py \\
        --images /path/to/images \\
        --annotations /path/to/annotations \\
        --output /path/to/output \\
        --bbox-format xyxy \\
        --category-name ball \\
        --train-ratio 0.7 --val-ratio 0.2 --test-ratio 0.1

If you don't want a split (single COCO file with everything),
pass --no-split.
"""

import argparse
import json
import random
import shutil
import sys
from pathlib import Path

try:
    from PIL import Image
except ImportError:
    Image = None


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ============================================================
# ARGUMENT PARSING
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert a flat image+annotation dataset into COCO format.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--images", "-i", required=True, type=Path,
        help="Path to the folder containing images (flat, no subfolders).",
    )
    parser.add_argument(
        "--annotations", "-a", required=True, type=Path,
        help="Path to the folder containing JSON annotations (flat, same filenames as images).",
    )
    parser.add_argument(
        "--output", "-o", required=True, type=Path,
        help="Path to the output folder where the COCO dataset will be created.",
    )
    parser.add_argument(
        "--bbox-format", choices=["xyxy", "xywh"], default="xyxy",
        help="Format of the 'rect' field in the source annotations. "
             "xyxy = [x1,y1,x2,y2], xywh = [x,y,width,height].",
    )
    parser.add_argument(
        "--category-name", default="ball",
        help="Name of the (single) object category to output.",
    )
    parser.add_argument(
        "--category-id", type=int, default=1,
        help="Category id to use in the COCO output.",
    )
    parser.add_argument(
        "--no-split", action="store_true",
        help="Do not split into train/val/test — write a single "
             "'all' split containing every image.",
    )
    parser.add_argument(
        "--train-ratio", type=float, default=0.70,
        help="Fraction of images to put in the train split.",
    )
    parser.add_argument(
        "--val-ratio", type=float, default=0.20,
        help="Fraction of images to put in the val split.",
    )
    parser.add_argument(
        "--test-ratio", type=float, default=0.10,
        help="Fraction of images to put in the test split.",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed used for the split, so results are reproducible.",
    )

    args = parser.parse_args()

    if not args.no_split:
        total = args.train_ratio + args.val_ratio + args.test_ratio
        if abs(total - 1.0) > 1e-6:
            parser.error(
                f"--train-ratio + --val-ratio + --test-ratio must equal 1.0 "
                f"(got {total})"
            )

    return args


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def convert_xyxy_to_xywh(rect):
    if not isinstance(rect, list) or len(rect) != 4:
        raise ValueError(f"Bounding box must be a list of 4 values: {rect}")

    x1, y1, x2, y2 = rect
    width = x2 - x1
    height = y2 - y1

    if width < 0 or height < 0:
        raise ValueError(f"Invalid XYXY bounding box: {rect}")

    return [x1, y1, width, height]


def get_normalized_bbox(rect, bbox_format):
    if bbox_format == "xyxy":
        return convert_xyxy_to_xywh(rect)
    elif bbox_format == "xywh":
        return rect
    else:
        raise ValueError(f"Unknown bbox format: {bbox_format}")


def find_image_annotation_pairs(images_dir, annotations_dir):
    """
    Match every image in images_dir with a same-stem JSON file in
    annotations_dir. Flat folders only — no recursion into subfolders.
    """

    pairs = []
    missing_labels = []

    for image_path in sorted(images_dir.iterdir()):

        if not image_path.is_file():
            continue

        if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        label_path = annotations_dir / f"{image_path.stem}.json"

        if not label_path.exists():
            missing_labels.append(image_path.name)
            continue

        pairs.append((image_path, label_path))

    return pairs, missing_labels


def load_annotation(label_path):
    with open(label_path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_image_dimensions(annotation_data, image_path):
    """
    Prefer the 'dimensions': [height, width] field from the
    annotation JSON. Fall back to reading the actual image file.
    """

    dimensions = annotation_data.get("dimensions")

    if dimensions is not None and len(dimensions) == 2:
        height = int(dimensions[0])
        width = int(dimensions[1])
        return width, height

    if Image is None:
        raise RuntimeError(
            "Pillow is required to read image dimensions when "
            "'dimensions' is missing from the annotation. "
            "Install it with: pip install pillow"
        )

    with Image.open(image_path) as im:
        return im.size  # (width, height)


# ============================================================
# COCO BUILDING
# ============================================================

def build_coco(pairs, bbox_format, category_name, category_id,
                split_name, output_root):
    """
    Copy images + build a COCO json for one split.
    """

    images_output_dir = output_root / "images" / split_name
    annotations_output_dir = output_root / "annotations"

    images_output_dir.mkdir(parents=True, exist_ok=True)
    annotations_output_dir.mkdir(parents=True, exist_ok=True)

    coco = {
        "info": {"description": "COCO Dataset", "version": "1.0"},
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": [
            {"id": category_id, "name": category_name, "supercategory": "object"}
        ],
    }

    annotation_id = 1
    successful_images = 0
    skipped_images = 0

    for image_id, (image_path, label_path) in enumerate(pairs, start=1):

        try:
            annotation_data = load_annotation(label_path)
        except Exception as e:
            print(f"[ERROR] Could not read {label_path}: {e}")
            skipped_images += 1
            continue

        try:
            width, height = get_image_dimensions(annotation_data, image_path)
        except Exception as e:
            print(f"[ERROR] Could not determine dimensions for {image_path}: {e}")
            skipped_images += 1
            continue

        destination = images_output_dir / image_path.name

        try:
            shutil.copy2(image_path, destination)
        except Exception as e:
            print(f"[ERROR] Could not copy {image_path}: {e}")
            skipped_images += 1
            continue

        coco["images"].append({
            "id": image_id,
            "file_name": image_path.name,
            "width": width,
            "height": height,
        })

        data = annotation_data.get("data", {})
        objects = data.get(category_name, [])

        for obj in objects:
            entire = obj.get("entire", {})
            rect = entire.get("rect")

            if rect is None:
                continue

            try:
                x, y, w, h = get_normalized_bbox(rect, bbox_format)
            except Exception as e:
                print(f"[WARNING] Invalid bbox {rect} in {label_path}: {e}")
                continue

            if w <= 0 or h <= 0:
                print(f"[WARNING] Invalid bbox dimensions {rect} in {label_path}")
                continue

            coco["annotations"].append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": category_id,
                "bbox": [float(x), float(y), float(w), float(h)],
                "area": float(w) * float(h),
                "iscrowd": 0,
            })
            annotation_id += 1

        successful_images += 1

    json_path = annotations_output_dir / f"instances_{split_name}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(coco, f, indent=2)

    print(f"\n[{split_name}] images copied: {successful_images}, "
          f"skipped: {skipped_images}, annotations: {len(coco['annotations'])}")
    print(f"[{split_name}] saved: {json_path}")

    return coco


# ============================================================
# MAIN
# ============================================================

def main():
    args = parse_args()

    images_dir = args.images
    annotations_dir = args.annotations
    output_root = args.output

    if not images_dir.exists():
        sys.exit(f"ERROR: images folder does not exist: {images_dir}")

    if not annotations_dir.exists():
        sys.exit(f"ERROR: annotations folder does not exist: {annotations_dir}")

    output_root.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("COCO CONVERSION")
    print("=" * 70)
    print(f"Images      : {images_dir}")
    print(f"Annotations : {annotations_dir}")
    print(f"Output      : {output_root}")
    print(f"BBox format : {args.bbox_format}")
    print(f"Category    : {args.category_name} (id={args.category_id})")

    pairs, missing_labels = find_image_annotation_pairs(images_dir, annotations_dir)

    if missing_labels:
        print(f"\n[WARNING] {len(missing_labels)} image(s) had no matching annotation file:")
        for name in missing_labels[:20]:
            print(f"    - {name}")
        if len(missing_labels) > 20:
            print(f"    ... and {len(missing_labels) - 20} more")

    if not pairs:
        sys.exit("\nERROR: No matching image/annotation pairs found. Nothing to convert.")

    print(f"\nTotal matched image-annotation pairs: {len(pairs)}")

    if args.no_split:
        build_coco(
            pairs, args.bbox_format, args.category_name, args.category_id,
            "all", output_root,
        )
    else:
        random.seed(args.seed)
        shuffled = pairs[:]
        random.shuffle(shuffled)

        total = len(shuffled)
        train_count = int(total * args.train_ratio)
        val_count = int(total * args.val_ratio)

        train_pairs = shuffled[:train_count]
        val_pairs = shuffled[train_count:train_count + val_count]
        test_pairs = shuffled[train_count + val_count:]

        print("\nSplit sizes:")
        print(f"  Train : {len(train_pairs)} ({len(train_pairs)/total*100:.2f}%)")
        print(f"  Val   : {len(val_pairs)} ({len(val_pairs)/total*100:.2f}%)")
        print(f"  Test  : {len(test_pairs)} ({len(test_pairs)/total*100:.2f}%)")

        build_coco(train_pairs, args.bbox_format, args.category_name,
                   args.category_id, "train", output_root)
        build_coco(val_pairs, args.bbox_format, args.category_name,
                   args.category_id, "val", output_root)
        build_coco(test_pairs, args.bbox_format, args.category_name,
                   args.category_id, "test", output_root)

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)
    print(f"\nOutput structure:\n")
    print(f"{output_root}/")
    print("├── images/")
    if args.no_split:
        print("│   └── all/")
    else:
        print("│   ├── train/")
        print("│   ├── val/")
        print("│   └── test/")
    print("└── annotations/")
    if args.no_split:
        print("    └── instances_all.json")
    else:
        print("    ├── instances_train.json")
        print("    ├── instances_val.json")
        print("    └── instances_test.json")


if __name__ == "__main__":
    main()
