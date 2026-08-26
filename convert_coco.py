import json
import random
import shutil
from pathlib import Path
from collections import defaultdict

# ============================================================
# CONFIGURATION
# ============================================================

# Your dataset root
SOURCE_ROOT = Path("/home/eng_megha/balldataset")

# Where the final COCO dataset will be created
OUTPUT_ROOT = Path("/home/eng_megha/balldataset_coco")

# Only these folders will be used
SELECTED_CASES = [
    "AI_train_caseB",
    "basic_train_caseA",
    "cc0_train_caseB",
    "match_train_caseA",
]

# These are explicitly excluded
EXCLUDED_NAMES = {
    "aug",
    "other",
    "volleyball",
}

# Split ratios
TRAIN_RATIO = 0.70
VAL_RATIO = 0.20
TEST_RATIO = 0.10

# Reproducible split
RANDOM_SEED = 42

# Image extensions
IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}

# COCO category
CATEGORIES = [
    {
        "id": 1,
        "name": "ball",
        "supercategory": "object"
    }
]


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def is_excluded(path: Path):
    """
    Check whether any folder in the path is one of the
    explicitly excluded folders.
    """
    return any(part.lower() in EXCLUDED_NAMES for part in path.parts)


def find_image_label_pairs():
    """
    Find images inside imgs/ folders and their corresponding
    JSON annotations inside sibling labels/ folders.
    """

    pairs = []
    missing_labels = []

    for case_name in SELECTED_CASES:

        case_root = SOURCE_ROOT / case_name

        if not case_root.exists():
            print(f"[WARNING] Folder not found: {case_root}")
            continue

        print(f"\nScanning: {case_root}")

        # Find all directories named imgs
        imgs_dirs = [
            p for p in case_root.rglob("imgs")
            if p.is_dir() and not is_excluded(p)
        ]

        for imgs_dir in imgs_dirs:

            # labels should be next to imgs
            labels_dir = imgs_dir.parent / "labels"

            if not labels_dir.exists():
                print(f"[WARNING] Labels directory missing:")
                print(f"          {labels_dir}")
                continue

            print(f"  Images : {imgs_dir}")
            print(f"  Labels : {labels_dir}")

            for image_path in sorted(imgs_dir.iterdir()):

                if not image_path.is_file():
                    continue

                if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                    continue

                # Same filename stem:
                # image1.jpg -> image1.json
                label_path = labels_dir / f"{image_path.stem}.json"

                if not label_path.exists():

                    missing_labels.append(image_path)

                    print(
                        f"    [MISSING LABEL] "
                        f"{image_path.name}"
                    )

                    continue

                pairs.append({
                    "image": image_path,
                    "label": label_path,
                    "case": case_name,
                })

    return pairs, missing_labels


def read_annotation(label_path):
    """
    Read your custom annotation format.

    Example:

    {
        "file_name": "image1.jpg",
        "dimensions": [848, 1264],
        "data": {
            "ball": [
                {
                    "entire": {
                        "rect": [477, 348, 546, 417]
                    }
                }
            ]
        }
    }
    """

    with open(label_path, "r", encoding="utf-8") as f:
        return json.load(f)


def convert_to_coco_bbox(rect):
    """
    Convert:

        [x1, y1, x2, y2]

    into COCO:

        [x, y, width, height]
    """

    if len(rect) != 4:
        raise ValueError(
            f"Invalid rect: {rect}"
        )

    x1, y1, x2, y2 = rect

    width = x2 - x1
    height = y2 - y1

    # Prevent negative boxes
    width = max(0, width)
    height = max(0, height)

    return [
        float(x1),
        float(y1),
        float(width),
        float(height),
    ]


def create_coco_dataset(items, split_name, output_dir):
    """
    Create COCO JSON and copy images for one split.
    """

    image_output_dir = output_dir / "images" / split_name
    annotation_output_dir = output_dir / "annotations"

    image_output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    annotation_output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    coco = {
        "info": {
            "description": "Ball Detection Dataset",
            "version": "1.0"
        },
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": CATEGORIES,
    }

    annotation_id = 1

    for image_id, item in enumerate(items, start=1):

        image_path = item["image"]
        label_path = item["label"]

        try:
            annotation_data = read_annotation(label_path)

        except Exception as e:
            print(
                f"[ERROR] Cannot read {label_path}: {e}"
            )
            continue

        # ----------------------------------------------------
        # Dimensions
        # ----------------------------------------------------

        dimensions = annotation_data.get("dimensions")

        if dimensions is not None and len(dimensions) == 2:

            # Your format appears to be:
            # [height, width]
            height = int(dimensions[0])
            width = int(dimensions[1])

        else:
            # Fallback: read actual image dimensions
            from PIL import Image

            with Image.open(image_path) as im:
                width, height = im.size

        # ----------------------------------------------------
        # Create unique output filename
        # ----------------------------------------------------

        # Avoid collisions such as:
        # basketball/img1.jpg
        # rugby/img1.jpg
        #
        # We prefix the case name.

        unique_name = (
            f"{item['case']}_"
            f"{image_path.parent.parent.name}_"
            f"{image_path.name}"
        )

        # Replace problematic characters
        unique_name = unique_name.replace("/", "_")
        unique_name = unique_name.replace("\\", "_")

        destination = image_output_dir / unique_name

        shutil.copy2(
            image_path,
            destination
        )

        # ----------------------------------------------------
        # COCO IMAGE ENTRY
        # ----------------------------------------------------

        coco["images"].append({
            "id": image_id,
            "file_name": unique_name,
            "width": width,
            "height": height,
        })

        # ----------------------------------------------------
        # Read BALL annotations
        # ----------------------------------------------------

        data = annotation_data.get("data", {})

        balls = data.get("ball", [])

        for ball in balls:

            entire = ball.get("entire", {})

            rect = entire.get("rect")

            if rect is None:
                continue

            try:
                bbox = convert_to_coco_bbox(rect)
            except Exception as e:
                print(
                    f"[WARNING] Bad bbox in {label_path}: {e}"
                )
                continue

            x, y, bbox_width, bbox_height = bbox

            # Ignore invalid boxes
            if bbox_width <= 0 or bbox_height <= 0:
                print(
                    f"[WARNING] Invalid bbox "
                    f"in {label_path}: {rect}"
                )
                continue

            # ------------------------------------------------
            # COCO ANNOTATION
            # ------------------------------------------------

            coco["annotations"].append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": 1,
                "bbox": bbox,
                "area": bbox_width * bbox_height,
                "iscrowd": 0,
            })

            annotation_id += 1

    # --------------------------------------------------------
    # Save JSON
    # --------------------------------------------------------

    json_path = (
        annotation_output_dir /
        f"instances_{split_name}.json"
    )

    with open(
        json_path,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            coco,
            f,
            indent=2
        )

    print(
        f"\nCreated {split_name}:"
    )
    print(
        f"  Images      : {len(coco['images'])}"
    )
    print(
        f"  Annotations : {len(coco['annotations'])}"
    )
    print(
        f"  JSON        : {json_path}"
    )

    return coco


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("COCO DATASET CREATION")
    print("=" * 70)

    print("\nSource:", SOURCE_ROOT)
    print("Output:", OUTPUT_ROOT)

    print("\nSelected cases:")

    for case in SELECTED_CASES:
        print(f"  + {case}")

    print("\nExcluded:")
    print("  - aug")
    print("  - other")
    print("  - volleyball")

    # --------------------------------------------------------
    # Find image/annotation pairs
    # --------------------------------------------------------

    pairs, missing_labels = find_image_label_pairs()

    print("\n" + "=" * 70)
    print("DATASET SUMMARY")
    print("=" * 70)

    print(f"Total image-label pairs: {len(pairs)}")
    print(f"Images without labels  : {len(missing_labels)}")

    if len(pairs) == 0:
        print("\nERROR: No image-label pairs found.")
        print(
            "Check SOURCE_ROOT and your imgs/labels structure."
        )
        return

    # --------------------------------------------------------
    # Shuffle
    # --------------------------------------------------------

    random.seed(RANDOM_SEED)

    random.shuffle(pairs)

    # --------------------------------------------------------
    # Calculate split sizes
    # --------------------------------------------------------

    total = len(pairs)

    train_count = int(total * TRAIN_RATIO)
    val_count = int(total * VAL_RATIO)

    # Remaining images go to test
    test_count = (
        total -
        train_count -
        val_count
    )

    train_items = pairs[:train_count]

    val_items = pairs[
        train_count:
        train_count + val_count
    ]

    test_items = pairs[
        train_count + val_count:
    ]

    print("\nSplit:")
    print(
        f"  Train: {len(train_items)} "
        f"({len(train_items) / total * 100:.2f}%)"
    )
    print(
        f"  Val  : {len(val_items)} "
        f"({len(val_items) / total * 100:.2f}%)"
    )
    print(
        f"  Test : {len(test_items)} "
        f"({len(test_items) / total * 100:.2f}%)"
    )

    # --------------------------------------------------------
    # Create output directory
    # --------------------------------------------------------

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Create COCO datasets
    # --------------------------------------------------------

    create_coco_dataset(
        train_items,
        "train",
        OUTPUT_ROOT
    )

    create_coco_dataset(
        val_items,
        "val",
        OUTPUT_ROOT
    )

    create_coco_dataset(
        test_items,
        "test",
        OUTPUT_ROOT
    )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)

    print("\nCOCO dataset created at:")

    print(OUTPUT_ROOT)

    print("\nFinal structure:")

    print(
        f"""
{OUTPUT_ROOT}/
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


if __name__ == "__main__":
    main()
