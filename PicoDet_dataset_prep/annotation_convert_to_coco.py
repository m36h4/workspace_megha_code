```python
import argparse
import json
import random
import shutil
from pathlib import Path

from PIL import Image


# ============================================================
# GENERIC COCO DATASET CONVERTER
# ============================================================
#
# Expected input:
#
# image_dir/
#     image1.jpg
#     image2.jpg
#     image3.png
#
# annotation_dir/
#     image1.json
#     image2.json
#     image3.json
#
#
# Output:
#
# output_dir/
# ├── images/
# │   ├── train/
# │   ├── val/
# │   └── test/
# │
# └── annotations/
#     ├── instances_train.json
#     ├── instances_val.json
#     └── instances_test.json
#
#
# Usage:
#
# python coco_converter.py \
#     --image-dir /path/to/images \
#     --annotation-dir /path/to/annotations \
#     --output-dir /path/to/output
#
#
# For XYXY annotations:
#
# python coco_converter.py \
#     --image-dir /path/to/images \
#     --annotation-dir /path/to/annotations \
#     --output-dir /path/to/output \
#     --bbox-format xyxy
#
#
# For XYWH annotations:
#
# python coco_converter.py \
#     --image-dir /path/to/images \
#     --annotation-dir /path/to/annotations \
#     --output-dir /path/to/output \
#     --bbox-format xywh
#
# ============================================================


# ============================================================
# CONFIGURATION
# ============================================================

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}

TRAIN_RATIO = 0.70
VAL_RATIO = 0.20
TEST_RATIO = 0.10

RANDOM_SEED = 42

CATEGORY_ID = 1
CATEGORY_NAME = "ball"
CATEGORY_SUPERCLASS = "object"


# ============================================================
# ARGUMENTS
# ============================================================

def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Convert image + JSON annotations "
            "into a COCO dataset."
        )
    )

    parser.add_argument(
        "--image-dir",
        required=True,
        type=Path,
        help="Folder containing input images."
    )

    parser.add_argument(
        "--annotation-dir",
        required=True,
        type=Path,
        help=(
            "Folder containing JSON annotations. "
            "Annotation filename must match image filename."
        )
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Folder where the COCO dataset will be created."
    )

    parser.add_argument(
        "--bbox-format",
        choices=["xyxy", "xywh"],
        default="xyxy",
        help=(
            "Input bounding box format. "
            "xyxy = [x1,y1,x2,y2], "
            "xywh = [x,y,width,height]. "
            "Default: xyxy"
        )
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
        help="Random seed used for train/val/test split."
    )

    return parser.parse_args()


# ============================================================
# VALIDATE INPUT
# ============================================================

def validate_input(args):

    if not args.image_dir.exists():
        raise FileNotFoundError(
            f"Image directory does not exist:\n"
            f"{args.image_dir}"
        )

    if not args.image_dir.is_dir():
        raise NotADirectoryError(
            f"Image path is not a directory:\n"
            f"{args.image_dir}"
        )

    if not args.annotation_dir.exists():
        raise FileNotFoundError(
            f"Annotation directory does not exist:\n"
            f"{args.annotation_dir}"
        )

    if not args.annotation_dir.is_dir():
        raise NotADirectoryError(
            f"Annotation path is not a directory:\n"
            f"{args.annotation_dir}"
        )

    if not (
        0 < TRAIN_RATIO < 1
        and 0 < VAL_RATIO < 1
        and 0 < TEST_RATIO < 1
    ):
        raise ValueError(
            "Split ratios must be between 0 and 1."
        )

    if abs(
        TRAIN_RATIO +
        VAL_RATIO +
        TEST_RATIO -
        1.0
    ) > 1e-6:

        raise ValueError(
            "TRAIN_RATIO + VAL_RATIO + TEST_RATIO "
            "must equal 1.0"
        )


# ============================================================
# FIND IMAGES
# ============================================================

def find_images(image_dir):

    images = []

    for path in image_dir.rglob("*"):

        if not path.is_file():
            continue

        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        images.append(path)

    return sorted(images)


# ============================================================
# CONVERT BBOX
# ============================================================

def convert_xyxy_to_xywh(rect):

    if not isinstance(rect, list):
        raise ValueError(
            f"Bounding box must be a list: {rect}"
        )

    if len(rect) != 4:
        raise ValueError(
            f"Bounding box must contain 4 values: {rect}"
        )

    x1, y1, x2, y2 = map(float, rect)

    width = x2 - x1
    height = y2 - y1

    if width <= 0 or height <= 0:
        raise ValueError(
            f"Invalid XYXY bounding box: {rect}"
        )

    return [
        x1,
        y1,
        width,
        height,
    ]


# ============================================================
# EXTRACT BOUNDING BOXES
# ============================================================

def extract_bboxes(annotation, bbox_format):

    data = annotation.get("data", {})

    balls = data.get("ball", [])

    bboxes = []

    for ball in balls:

        entire = ball.get("entire", {})

        rect = entire.get("rect")

        if rect is None:
            continue

        if not isinstance(rect, list) or len(rect) != 4:

            print(
                f"  [WARNING] Invalid bbox skipped: "
                f"{rect}"
            )

            continue

        try:

            if bbox_format == "xyxy":

                bbox = convert_xyxy_to_xywh(rect)

            else:

                bbox = list(
                    map(float, rect)
                )

                if (
                    bbox[2] <= 0
                    or bbox[3] <= 0
                ):
                    raise ValueError(
                        f"Invalid XYWH bbox: {rect}"
                    )

            bboxes.append(bbox)

        except ValueError as e:

            print(
                f"  [WARNING] {e}"
            )

    return bboxes


# ============================================================
# GET IMAGE DIMENSIONS
# ============================================================

def get_image_dimensions(
    image_path,
    annotation,
):

    dimensions = annotation.get("dimensions")

    if (
        isinstance(dimensions, list)
        and len(dimensions) == 2
    ):

        # Expected annotation format:
        #
        # dimensions: [height, width]

        height = int(dimensions[0])
        width = int(dimensions[1])

        return width, height

    # --------------------------------------------------------
    # Fallback: read actual image dimensions
    # --------------------------------------------------------

    with Image.open(image_path) as image:

        width, height = image.size

    return width, height


# ============================================================
# FIND ANNOTATION
# ============================================================

def find_annotation(
    image_path,
    image_dir,
    annotation_dir,
):

    relative_path = image_path.relative_to(
        image_dir
    )

    # First try to preserve any nested structure.
    #
    # Example:
    #
    # images/a/image1.jpg
    # annotations/a/image1.json

    matching_path = (
        annotation_dir /
        relative_path.with_suffix(".json")
    )

    if matching_path.exists():
        return matching_path

    # --------------------------------------------------------
    # Fallback:
    #
    # Search recursively by filename.
    # --------------------------------------------------------

    matches = list(
        annotation_dir.rglob(
            f"{image_path.stem}.json"
        )
    )

    if len(matches) == 1:
        return matches[0]

    if len(matches) > 1:

        print(
            f"[WARNING] Multiple annotations found "
            f"for {image_path.name}"
        )

        print(
            f"          Using: {matches[0]}"
        )

        return matches[0]

    return None


# ============================================================
# CREATE UNIQUE IMAGE NAME
# ============================================================

def get_unique_image_name(
    image_path,
    image_dir,
):

    relative_path = image_path.relative_to(
        image_dir
    )

    # If image is directly inside image_dir:
    #
    # image1.jpg
    #
    # keep original name.
    #

    if len(relative_path.parts) == 1:
        return image_path.name

    # --------------------------------------------------------
    # For nested images, flatten the path.
    #
    # Example:
    #
    # images/
    #   class1/
    #       image1.jpg
    #
    # becomes:
    #
    # class1_image1.jpg
    # --------------------------------------------------------

    parts = list(
        relative_path.parts
    )

    filename = parts[-1]

    directory_parts = parts[:-1]

    prefix = "_".join(
        directory_parts
    )

    return f"{prefix}_{filename}"


# ============================================================
# COLLECT IMAGE / ANNOTATION PAIRS
# ============================================================

def collect_pairs(
    image_dir,
    annotation_dir,
):

    print()
    print("=" * 70)
    print("SEARCHING FOR IMAGES AND ANNOTATIONS")
    print("=" * 70)

    images = find_images(
        image_dir
    )

    if not images:

        raise RuntimeError(
            "No images found in:\n"
            f"{image_dir}"
        )

    pairs = []

    missing_annotations = 0

    for image_path in images:

        annotation_path = find_annotation(
            image_path,
            image_dir,
            annotation_dir,
        )

        if annotation_path is None:

            print(
                f"[MISSING LABEL] "
                f"{image_path.name}"
            )

            missing_annotations += 1

            continue

        pairs.append({
            "image": image_path,
            "annotation": annotation_path,
        })

    print()
    print(
        f"Images found          : {len(images)}"
    )

    print(
        f"Valid image-label pairs: {len(pairs)}"
    )

    print(
        f"Missing annotations    : "
        f"{missing_annotations}"
    )

    if not pairs:

        raise RuntimeError(
            "No valid image/annotation pairs found."
        )

    return pairs


# ============================================================
# CREATE COCO DATASET
# ============================================================

def create_coco_dataset(
    pairs,
    split_name,
    args,
):

    print()
    print("=" * 70)
    print(
        f"CREATING COCO: {split_name.upper()}"
    )
    print("=" * 70)

    # --------------------------------------------------------
    # Output directories
    #
    # IMPORTANT:
    #
    # There are NO dataset/case folders underneath.
    # --------------------------------------------------------

    images_output_dir = (
        args.output_dir /
        "images" /
        split_name
    )

    annotations_output_dir = (
        args.output_dir /
        "annotations"
    )

    images_output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    annotations_output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # COCO structure
    # --------------------------------------------------------

    coco = {

        "info": {
            "description":
                "COCO Dataset",
            "version":
                "1.0",
        },

        "licenses": [],

        "images": [],

        "annotations": [],

        "categories": [
            {
                "id": CATEGORY_ID,
                "name": CATEGORY_NAME,
                "supercategory":
                    CATEGORY_SUPERCLASS,
            }
        ],
    }

    annotation_id = 1

    copied_images = 0
    skipped_images = 0

    used_filenames = set()

    # --------------------------------------------------------
    # Process images
    # --------------------------------------------------------

    for image_id, pair in enumerate(
        pairs,
        start=1
    ):

        image_path = pair["image"]

        annotation_path = pair[
            "annotation"
        ]

        print(
            f"[{image_id}/{len(pairs)}] "
            f"{image_path.name}"
        )

        # ----------------------------------------------------
        # Read JSON
        # ----------------------------------------------------

        try:

            with open(
                annotation_path,
                "r",
                encoding="utf-8"
            ) as file:

                annotation = json.load(file)

        except Exception as e:

            print(
                f"  [ERROR] Could not read annotation:"
            )

            print(
                f"          {annotation_path}"
            )

            print(
                f"          {e}"
            )

            skipped_images += 1

            continue

        # ----------------------------------------------------
        # Get image size
        # ----------------------------------------------------

        try:

            width, height = (
                get_image_dimensions(
                    image_path,
                    annotation,
                )
            )

        except Exception as e:

            print(
                f"  [ERROR] Could not determine "
                f"image dimensions:"
            )

            print(
                f"          {e}"
            )

            skipped_images += 1

            continue

        # ----------------------------------------------------
        # Generate flattened filename
        # ----------------------------------------------------

        unique_name = get_unique_image_name(
            image_path,
            args.image_dir,
        )

        # ----------------------------------------------------
        # Prevent filename collision
        # ----------------------------------------------------

        if unique_name in used_filenames:

            stem = Path(
                unique_name
            ).stem

            suffix = Path(
                unique_name
            ).suffix

            counter = 2

            while (
                f"{stem}_{counter}{suffix}"
                in used_filenames
            ):

                counter += 1

            unique_name = (
                f"{stem}_{counter}{suffix}"
            )

        used_filenames.add(
            unique_name
        )

        destination = (
            images_output_dir /
            unique_name
        )

        # ----------------------------------------------------
        # Copy image
        # ----------------------------------------------------

        try:

            shutil.copy2(
                image_path,
                destination
            )

        except Exception as e:

            print(
                f"  [ERROR] Could not copy image:"
            )

            print(
                f"          {e}"
            )

            skipped_images += 1

            continue

        # ----------------------------------------------------
        # Add image to COCO
        # ----------------------------------------------------

        coco["images"].append({

            "id":
                image_id,

            "file_name":
                unique_name,

            "width":
                width,

            "height":
                height,
        })

        # ----------------------------------------------------
        # Extract bounding boxes
        # ----------------------------------------------------

        bboxes = extract_bboxes(
            annotation,
            args.bbox_format,
        )

        # ----------------------------------------------------
        # Add annotations
        # ----------------------------------------------------

        for bbox in bboxes:

            x = bbox[0]
            y = bbox[1]
            bbox_width = bbox[2]
            bbox_height = bbox[3]

            # ------------------------------------------------
            # Optional clipping
            #
            # Prevent boxes from extending outside image.
            # ------------------------------------------------

            x = max(
                0,
                min(x, width)
            )

            y = max(
                0,
                min(y, height)
            )

            bbox_width = min(
                bbox_width,
                width - x
            )

            bbox_height = min(
                bbox_height,
                height - y
            )

            if (
                bbox_width <= 0
                or bbox_height <= 0
            ):
                print(
                    "  [WARNING] Bbox became invalid "
                    "after clipping. Skipped."
                )

                continue

            coco["annotations"].append({

                "id":
                    annotation_id,

                "image_id":
                    image_id,

                "category_id":
                    CATEGORY_ID,

                "bbox": [
                    x,
                    y,
                    bbox_width,
                    bbox_height,
                ],

                "area":
                    bbox_width *
                    bbox_height,

                "iscrowd":
                    0,
            })

            annotation_id += 1

        copied_images += 1

    # --------------------------------------------------------
    # Save COCO JSON
    # --------------------------------------------------------

    json_path = (
        annotations_output_dir /
        f"instances_{split_name}.json"
    )

    with open(
        json_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            coco,
            file,
            indent=2
        )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print()
    print("-" * 70)

    print(
        f"Images copied       : "
        f"{copied_images}"
    )

    print(
        f"Images skipped      : "
        f"{skipped_images}"
    )

    print(
        f"COCO images         : "
        f"{len(coco['images'])}"
    )

    print(
        f"COCO annotations    : "
        f"{len(coco['annotations'])}"
    )

    print(
        f"JSON saved          : "
        f"{json_path}"
    )

    return coco


# ============================================================
# MAIN
# ============================================================

def main():

    args = parse_arguments()

    # --------------------------------------------------------
    # Validate input
    # --------------------------------------------------------

    validate_input(args)

    print()
    print("=" * 70)
    print("GENERIC COCO DATASET CONVERTER")
    print("=" * 70)

    print()
    print(
        f"Image directory      : "
        f"{args.image_dir}"
    )

    print(
        f"Annotation directory : "
        f"{args.annotation_dir}"
    )

    print(
        f"Output directory     : "
        f"{args.output_dir}"
    )

    print(
        f"Bounding box format  : "
        f"{args.bbox_format.upper()}"
    )

    print(
        f"Random seed          : "
        f"{args.seed}"
    )

    print()
    print(
        "Split ratio:"
    )

    print(
        f"  Train = {TRAIN_RATIO:.0%}"
    )

    print(
        f"  Val   = {VAL_RATIO:.0%}"
    )

    print(
        f"  Test  = {TEST_RATIO:.0%}"
    )

    # --------------------------------------------------------
    # Collect pairs
    # --------------------------------------------------------

    pairs = collect_pairs(
        args.image_dir,
        args.annotation_dir,
    )

    # --------------------------------------------------------
    # Random split
    # --------------------------------------------------------

    random.seed(
        args.seed
    )

    random.shuffle(
        pairs
    )

    total = len(pairs)

    train_count = int(
        total *
        TRAIN_RATIO
    )

    val_count = int(
        total *
        VAL_RATIO
    )

    train_pairs = pairs[
        :train_count
    ]

    val_pairs = pairs[
        train_count:
        train_count +
        val_count
    ]

    test_pairs = pairs[
        train_count +
        val_count:
    ]

    # --------------------------------------------------------
    # Print split
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("DATASET SPLIT")
    print("=" * 70)

    print()
    print(
        f"Total : {total}"
    )

    print(
        f"Train : {len(train_pairs)} "
        f"({len(train_pairs) / total:.2%})"
    )

    print(
        f"Val   : {len(val_pairs)} "
        f"({len(val_pairs) / total:.2%})"
    )

    print(
        f"Test  : {len(test_pairs)} "
        f"({len(test_pairs) / total:.2%})"
    )

    # --------------------------------------------------------
    # Create COCO train
    # --------------------------------------------------------

    create_coco_dataset(
        train_pairs,
        "train",
        args,
    )

    # --------------------------------------------------------
    # Create COCO validation
    # --------------------------------------------------------

    create_coco_dataset(
        val_pairs,
        "val",
        args,
    )

    # --------------------------------------------------------
    # Create COCO test
    # --------------------------------------------------------

    create_coco_dataset(
        test_pairs,
        "test",
        args,
    )

    # --------------------------------------------------------
    # Final output
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("CONVERSION COMPLETE")
    print("=" * 70)

    print()
    print(
        "COCO dataset created at:"
    )

    print(
        f"  {args.output_dir}"
    )

    print()
    print(
        "Output structure:"
    )

    print(
        f"""
{args.output_dir}/
│
├── images/
│   ├── train/
│   ├── val/
│   └── test/
│
└── annotations/
    ├── instances_train.json
    ├── instances_val.json
    └── instances_test.json
"""
    )

    print(
        "Original images and annotations were "
        "not modified."
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()
```
