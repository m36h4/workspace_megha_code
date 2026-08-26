import json
import random
import shutil
from pathlib import Path
from collections import Counter
from PIL import Image


# ============================================================
# CONFIGURATION
# ============================================================

# ------------------------------------------------------------
# CHANGE THESE TWO PATHS IF NEEDED
# ------------------------------------------------------------

SOURCE_ROOT = Path("/home/eng_megha/balldataset")

OUTPUT_ROOT = Path("/home/eng_megha/balldataset_coco")


# ------------------------------------------------------------
# ONLY THESE CASES WILL BE USED
# ------------------------------------------------------------

SELECTED_CASES = [
    "AI_train_caseB",
    "basic_train_caseA",
    "cc0_train_caseB",
    "match_train_caseA",
]


# ------------------------------------------------------------
# THESE FOLDERS WILL NEVER BE USED
# ------------------------------------------------------------

EXCLUDED_FOLDERS = {
    "aug",
    "other",
    "volleyball",
}


# ------------------------------------------------------------
# SPLIT
# ------------------------------------------------------------

TRAIN_RATIO = 0.70
VAL_RATIO = 0.20
TEST_RATIO = 0.10


# ------------------------------------------------------------
# RANDOM SEED
# ------------------------------------------------------------

RANDOM_SEED = 42


# ------------------------------------------------------------
# IMAGE TYPES
# ------------------------------------------------------------

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
    ".JPG",
    ".JPEG",
    ".PNG",
}


# ------------------------------------------------------------
# COCO CATEGORY
# ------------------------------------------------------------

CATEGORIES = [
    {
        "id": 1,
        "name": "ball",
        "supercategory": "object",
    }
]


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def is_excluded_path(path):
    """
    Returns True if any folder in the path is explicitly
    excluded.

    Examples:
        aug
        other
        volleyball
    """

    for part in path.parts:

        if part.lower() in EXCLUDED_FOLDERS:
            return True

    return False


# ------------------------------------------------------------

def find_image_label_pairs():
    """
    Find all:

        imgs/
        labels/

    pairs inside the selected case folders.
    """

    pairs = []

    missing_labels = []

    print("\n" + "=" * 70)
    print("SEARCHING FOR IMAGES AND LABELS")
    print("=" * 70)

    for case_name in SELECTED_CASES:

        case_root = SOURCE_ROOT / case_name

        if not case_root.exists():

            print(
                f"\n[WARNING] Case folder does not exist:"
                f"\n          {case_root}"
            )

            continue

        print(f"\nCASE: {case_name}")

        # Find all imgs directories recursively
        imgs_dirs = []

        for path in case_root.rglob("imgs"):

            if not path.is_dir():
                continue

            if is_excluded_path(path):
                continue

            imgs_dirs.append(path)

        print(
            f"  Found {len(imgs_dirs)} imgs directories"
        )

        for imgs_dir in sorted(imgs_dirs):

            labels_dir = imgs_dir.parent / "labels"

            if not labels_dir.exists():

                print(
                    f"\n  [WARNING] Labels directory missing:"
                    f"\n            {labels_dir}"
                )

                continue

            for image_path in sorted(imgs_dir.iterdir()):

                if not image_path.is_file():
                    continue

                if image_path.suffix.lower() not in {
                    ".jpg",
                    ".jpeg",
                    ".png",
                    ".bmp",
                    ".webp",
                }:
                    continue

                # ------------------------------------------------
                # Find corresponding JSON
                # ------------------------------------------------

                label_path = (
                    labels_dir /
                    f"{image_path.stem}.json"
                )

                if not label_path.exists():

                    missing_labels.append(image_path)

                    print(
                        f"  [MISSING LABEL] "
                        f"{image_path}"
                    )

                    continue

                pairs.append(
                    {
                        "image": image_path,
                        "label": label_path,
                        "case": case_name,
                    }
                )

    return pairs, missing_labels


# ------------------------------------------------------------

def load_json(json_path):
    """
    Load one annotation JSON.
    """

    try:

        with open(
            json_path,
            "r",
            encoding="utf-8"
        ) as f:

            return json.load(f)

    except Exception as e:

        print(
            f"\n[ERROR] Could not read JSON:"
            f"\n        {json_path}"
            f"\n        {e}"
        )

        return None


# ------------------------------------------------------------

def get_image_dimensions(image_path):
    """
    Read the actual pixel dimensions from the image.

    Returns:
        width, height
    """

    try:

        with Image.open(image_path) as img:

            width, height = img.size

        return width, height

    except Exception as e:

        print(
            f"\n[ERROR] Could not open image:"
            f"\n        {image_path}"
            f"\n        {e}"
        )

        return None, None


# ------------------------------------------------------------

def get_annotation_dimensions(annotation):
    """
    Your JSON format is:

        "dimensions": [
            576,
            384
        ]

    We interpret this as:

        width  = 576
        height = 384
    """

    dimensions = annotation.get("dimensions")

    if (
        not isinstance(dimensions, list)
        or len(dimensions) != 2
    ):

        return None, None

    width = int(dimensions[0])
    height = int(dimensions[1])

    return width, height


# ------------------------------------------------------------

def convert_bbox_to_coco(rect):
    """
    IMPORTANT:

    Your dataset rect is:

        [x, y, width, height]

    COCO uses exactly the same representation:

        [x, y, width, height]

    Therefore NO subtraction is performed.
    """

    if (
        not isinstance(rect, list)
        or len(rect) != 4
    ):

        return None

    x = float(rect[0])
    y = float(rect[1])
    width = float(rect[2])
    height = float(rect[3])

    return [
        x,
        y,
        width,
        height,
    ]


# ------------------------------------------------------------

def validate_bbox(
    bbox,
    image_width,
    image_height,
    image_name,
):
    """
    Validate COCO bbox.

    Expected:

        x >= 0
        y >= 0
        width > 0
        height > 0
        x + width <= image_width
        y + height <= image_height
    """

    if bbox is None:

        return False

    x, y, width, height = bbox

    # --------------------------------------------------------
    # Basic checks
    # --------------------------------------------------------

    if width <= 0 or height <= 0:

        print(
            f"\n[INVALID BBOX] {image_name}"
            f"\n  bbox = {bbox}"
            f"\n  Reason: width/height <= 0"
        )

        return False

    if x < 0 or y < 0:

        print(
            f"\n[INVALID BBOX] {image_name}"
            f"\n  bbox = {bbox}"
            f"\n  Reason: negative x/y"
        )

        return False

    # --------------------------------------------------------
    # Boundary checks
    # --------------------------------------------------------

    right = x + width
    bottom = y + height

    if right > image_width:

        print(
            f"\n[INVALID BBOX] {image_name}"
            f"\n  bbox = {bbox}"
            f"\n  Image width = {image_width}"
            f"\n  x + width = {right}"
            f"\n  Reason: box extends beyond right edge"
        )

        return False

    if bottom > image_height:

        print(
            f"\n[INVALID BBOX] {image_name}"
            f"\n  bbox = {bbox}"
            f"\n  Image height = {image_height}"
            f"\n  y + height = {bottom}"
            f"\n  Reason: box extends beyond bottom edge"
        )

        return False

    return True


# ------------------------------------------------------------

def make_unique_filename(item):
    """
    Avoid filename collisions.

    Example:

        basketball/img001.jpg
        rugby/img001.jpg

    become different filenames.
    """

    case_name = item["case"]

    image_path = item["image"]

    # Get sport/category folder
    #
    # imgs/
    #   parent = basketball
    #
    category_name = image_path.parent.parent.name

    filename = image_path.name

    unique_name = (
        f"{case_name}_"
        f"{category_name}_"
        f"{filename}"
    )

    # Make filename safe
    unique_name = unique_name.replace(
        " ",
        "_"
    )

    unique_name = unique_name.replace(
        "/",
        "_"
    )

    unique_name = unique_name.replace(
        "\\",
        "_"
    )

    return unique_name


# ============================================================
# BUILD ONE COCO SPLIT
# ============================================================

def create_coco_split(
    items,
    split_name,
    output_root,
):
    """
    Create:

        images/train/
        images/val/
        images/test/

    and corresponding COCO JSON.
    """

    image_output_dir = (
        output_root /
        "images" /
        split_name
    )

    annotation_output_dir = (
        output_root /
        "annotations"
    )

    image_output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    annotation_output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # COCO structure
    # --------------------------------------------------------

    coco = {

        "info": {
            "description": "Ball Detection Dataset",
            "version": "1.0",
        },

        "licenses": [],

        "images": [],

        "annotations": [],

        "categories": CATEGORIES,
    }

    annotation_id = 1

    valid_images = 0

    invalid_boxes = 0

    copied_images = set()

    # --------------------------------------------------------
    # Process images
    # --------------------------------------------------------

    for image_id, item in enumerate(
        items,
        start=1
    ):

        image_path = item["image"]

        label_path = item["label"]

        annotation = load_json(
            label_path
        )

        if annotation is None:
            continue

        # ----------------------------------------------------
        # JSON dimensions
        # ----------------------------------------------------

        json_width, json_height = (
            get_annotation_dimensions(
                annotation
            )
        )

        if (
            json_width is None
            or json_height is None
        ):

            print(
                f"\n[WARNING] Invalid dimensions:"
                f"\n          {label_path}"
            )

            continue

        # ----------------------------------------------------
        # Actual image dimensions
        # ----------------------------------------------------

        actual_width, actual_height = (
            get_image_dimensions(
                image_path
            )
        )

        if (
            actual_width is None
            or actual_height is None
        ):

            continue

        # ----------------------------------------------------
        # Compare JSON and actual image dimensions
        # ----------------------------------------------------

        if (
            json_width != actual_width
            or json_height != actual_height
        ):

            print(
                "\n[DIMENSION MISMATCH]"
            )

            print(
                f"  Image : {image_path}"
            )

            print(
                f"  JSON  : "
                f"{json_width} x {json_height}"
            )

            print(
                f"  Actual: "
                f"{actual_width} x {actual_height}"
            )

            print(
                "  NOTE: Annotation coordinates "
                "will NOT be rotated automatically."
            )

        # ----------------------------------------------------
        # Use actual image dimensions for COCO
        # ----------------------------------------------------

        image_width = actual_width
        image_height = actual_height

        # ----------------------------------------------------
        # Unique filename
        # ----------------------------------------------------

        unique_name = make_unique_filename(
            item
        )

        destination = (
            image_output_dir /
            unique_name
        )

        # Prevent accidental duplicate filename
        if unique_name in copied_images:

            print(
                f"\n[ERROR] Duplicate output filename:"
                f"\n        {unique_name}"
            )

            continue

        copied_images.add(
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
                f"\n[ERROR] Could not copy:"
                f"\n        {image_path}"
                f"\n        {e}"
            )

            continue

        # ----------------------------------------------------
        # COCO image entry
        # ----------------------------------------------------

        coco["images"].append(
            {
                "id": image_id,
                "file_name": unique_name,
                "width": image_width,
                "height": image_height,
            }
        )

        valid_images += 1

        # ----------------------------------------------------
        # Get ball annotations
        # ----------------------------------------------------

        data = annotation.get(
            "data",
            {}
        )

        balls = data.get(
            "ball",
            []
        )

        if not isinstance(
            balls,
            list
        ):

            continue

        # ----------------------------------------------------
        # Process each ball
        # ----------------------------------------------------

        for ball in balls:

            if not isinstance(
                ball,
                dict
            ):
                continue

            entire = ball.get(
                "entire",
                {}
            )

            if not isinstance(
                entire,
                dict
            ):
                continue

            rect = entire.get(
                "rect"
            )

            if rect is None:
                continue

            # ------------------------------------------------
            # Convert
            # ------------------------------------------------

            bbox = convert_bbox_to_coco(
                rect
            )

            # ------------------------------------------------
            # Validate
            # ------------------------------------------------

            if not validate_bbox(
                bbox,
                image_width,
                image_height,
                image_path.name,
            ):

                invalid_boxes += 1

                continue

            x, y, width, height = bbox

            # ------------------------------------------------
            # COCO annotation
            # ------------------------------------------------

            coco["annotations"].append(
                {
                    "id": annotation_id,

                    "image_id": image_id,

                    "category_id": 1,

                    "bbox": [
                        x,
                        y,
                        width,
                        height,
                    ],

                    "area": (
                        width *
                        height
                    ),

                    "iscrowd": 0,
                }
            )

            annotation_id += 1

    # --------------------------------------------------------
    # Save COCO JSON
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

    # --------------------------------------------------------
    # Print statistics
    # --------------------------------------------------------

    print(
        "\n" + "-" * 70
    )

    print(
        f"CREATED: {split_name.upper()}"
    )

    print(
        "-" * 70
    )

    print(
        f"Images      : "
        f"{len(coco['images'])}"
    )

    print(
        f"Annotations : "
        f"{len(coco['annotations'])}"
    )

    print(
        f"Invalid boxes skipped: "
        f"{invalid_boxes}"
    )

    print(
        f"JSON:"
        f"\n  {json_path}"
    )

    return coco


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "\n" +
        "=" * 70
    )

    print(
        "COCO BALL DATASET CREATOR"
    )

    print(
        "=" * 70
    )

    print(
        f"\nSOURCE:"
        f"\n  {SOURCE_ROOT}"
    )

    print(
        f"\nOUTPUT:"
        f"\n  {OUTPUT_ROOT}"
    )

    # --------------------------------------------------------
    # Print selected folders
    # --------------------------------------------------------

    print(
        "\nINCLUDED CASES:"
    )

    for case in SELECTED_CASES:

        print(
            f"  + {case}"
        )

    print(
        "\nEXCLUDED FOLDERS:"
    )

    for folder in sorted(
        EXCLUDED_FOLDERS
    ):

        print(
            f"  - {folder}"
        )

    # --------------------------------------------------------
    # Check source
    # --------------------------------------------------------

    if not SOURCE_ROOT.exists():

        print(
            "\n[ERROR] SOURCE_ROOT does not exist:"
        )

        print(
            SOURCE_ROOT
        )

        return

    # --------------------------------------------------------
    # Find image/label pairs
    # --------------------------------------------------------

    pairs, missing_labels = (
        find_image_label_pairs()
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print(
        "\n" +
        "=" * 70
    )

    print(
        "DATASET DISCOVERY SUMMARY"
    )

    print(
        "=" * 70
    )

    print(
        f"Image-label pairs : "
        f"{len(pairs)}"
    )

    print(
        f"Missing labels    : "
        f"{len(missing_labels)}"
    )

    if len(pairs) == 0:

        print(
            "\n[ERROR] No image-label pairs found."
        )

        return

    # --------------------------------------------------------
    # Count by case
    # --------------------------------------------------------

    case_counts = Counter(
        item["case"]
        for item in pairs
    )

    print(
        "\nImages by case:"
    )

    for case_name in SELECTED_CASES:

        print(
            f"  {case_name}: "
            f"{case_counts.get(case_name, 0)}"
        )

    # --------------------------------------------------------
    # Shuffle
    # --------------------------------------------------------

    random.seed(
        RANDOM_SEED
    )

    random.shuffle(
        pairs
    )

    # --------------------------------------------------------
    # Split
    # --------------------------------------------------------

    total = len(pairs)

    train_count = int(
        total *
        TRAIN_RATIO
    )

    val_count = int(
        total *
        VAL_RATIO
    )

    # Everything remaining goes to test.
    # This guarantees exactly 100%.

    test_count = (
        total -
        train_count -
        val_count
    )

    train_items = pairs[
        :train_count
    ]

    val_items = pairs[
        train_count:
        train_count + val_count
    ]

    test_items = pairs[
        train_count + val_count:
    ]

    # --------------------------------------------------------
    # Print split
    # --------------------------------------------------------

    print(
        "\n" +
        "=" * 70
    )

    print(
        "DATASET SPLIT"
    )

    print(
        "=" * 70
    )

    print(
        f"Total : {total}"
    )

    print(
        f"Train : {len(train_items)} "
        f"({len(train_items) / total * 100:.2f}%)"
    )

    print(
        f"Val   : {len(val_items)} "
        f"({len(val_items) / total * 100:.2f}%)"
    )

    print(
        f"Test  : {len(test_items)} "
        f"({len(test_items) / total * 100:.2f}%)"
    )

    # --------------------------------------------------------
    # Create output directory
    # --------------------------------------------------------

    if OUTPUT_ROOT.exists():

        print(
            "\n[WARNING]"
        )

        print(
            "Output directory already exists:"
        )

        print(
            OUTPUT_ROOT
        )

        answer = input(
            "\nDelete and recreate it? "
            "[y/N]: "
        ).strip().lower()

        if answer == "y":

            shutil.rmtree(
                OUTPUT_ROOT
            )

        else:

            print(
                "\nStopped."
            )

            print(
                "Choose another OUTPUT_ROOT "
                "or delete the existing directory."
            )

            return

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Create TRAIN
    # --------------------------------------------------------

    create_coco_split(
        train_items,
        "train",
        OUTPUT_ROOT
    )

    # --------------------------------------------------------
    # Create VAL
    # --------------------------------------------------------

    create_coco_split(
        val_items,
        "val",
        OUTPUT_ROOT
    )

    # --------------------------------------------------------
    # Create TEST
    # --------------------------------------------------------

    create_coco_split(
        test_items,
        "test",
        OUTPUT_ROOT
    )

    # --------------------------------------------------------
    # Final structure
    # --------------------------------------------------------

    print(
        "\n" +
        "=" * 70
    )

    print(
        "DONE"
    )

    print(
        "=" * 70
    )

    print(
        f"""
COCO DATASET:

{OUTPUT_ROOT}/
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

    # --------------------------------------------------------
    # Final counts
    # --------------------------------------------------------

    print(
        "FINAL SPLIT:"
    )

    print(
        f"  Train: {len(train_items)}"
    )

    print(
        f"  Val  : {len(val_items)}"
    )

    print(
        f"  Test : {len(test_items)}"
    )

    print(
        f"  Total: {total}"
    )

    print(
        "\nRandom seed:"
        f" {RANDOM_SEED}"
    )

    print(
        "\nBBOX FORMAT USED:"
    )

    print(
        "  rect = [x, y, width, height]"
    )

    print(
        "  COCO = [x, y, width, height]"
    )

    print(
        "\nDIMENSIONS FORMAT USED:"
    )

    print(
        "  dimensions = [width, height]"
    )

    print(
        "\nNo automatic 90-degree rotation was applied."
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()
