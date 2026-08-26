import json
import random
import shutil
import csv
from pathlib import Path
from collections import Counter

from PIL import Image


# ============================================================
# CONFIGURATION
# ============================================================

SOURCE_ROOT = Path("/home/eng_megha/balldataset")

OUTPUT_ROOT = Path("/home/eng_megha/balldataset_coco")


# Only these folders
SELECTED_CASES = [
    "AI_train_caseB",
    "basic_train_caseA",
    "cc0_train_caseB",
    "match_train_caseA",
]


# Explicitly ignored folders
EXCLUDED_FOLDERS = {
    "aug",
    "other",
    "volleyball",
}


# Split
TRAIN_RATIO = 0.70
VAL_RATIO = 0.20
TEST_RATIO = 0.10


# Reproducibility
RANDOM_SEED = 42


# ============================================================
# IMPORTANT ANNOTATION SETTINGS
# ============================================================

# Your source annotation is assumed to be:
#
#     [x1, y1, x2, y2]
#
# NOT COCO format.
#
SOURCE_RECT_FORMAT = "xyxy"


# ------------------------------------------------------------
# Rotation handling
# ------------------------------------------------------------
#
# "auto" tries to determine whether the annotation appears
# to correspond to a rotated coordinate system.
#
# You can later force:
#
#     0
#     90
#     180
#     270
#
# if we determine the dataset's exact convention.
#
ROTATION = "auto"


# ------------------------------------------------------------
# IMPORTANT:
#
# Do not automatically accept a rotation just because it
# produces numbers inside the image.
#
# Auto mode will flag ambiguous cases for inspection.
# ------------------------------------------------------------


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


# ============================================================
# COCO CATEGORY
# ============================================================

CATEGORIES = [
    {
        "id": 1,
        "name": "ball",
        "supercategory": "object",
    }
]


# ============================================================
# PATH HELPERS
# ============================================================

def is_excluded(path):
    """
    Check whether any component of a path is excluded.
    """

    for part in path.parts:

        if part.lower() in EXCLUDED_FOLDERS:
            return True

    return False


# ============================================================
# FIND IMAGE + JSON PAIRS
# ============================================================

def find_pairs():

    pairs = []

    missing_labels = []

    print("\n" + "=" * 70)
    print("SEARCHING DATASET")
    print("=" * 70)

    for case_name in SELECTED_CASES:

        case_root = SOURCE_ROOT / case_name

        if not case_root.exists():

            print(
                f"\n[WARNING] Missing case:"
                f"\n{case_root}"
            )

            continue

        print(
            f"\nScanning: {case_name}"
        )

        imgs_dirs = [
            p
            for p in case_root.rglob("imgs")
            if p.is_dir()
            and not is_excluded(p)
        ]

        for imgs_dir in sorted(imgs_dirs):

            labels_dir = (
                imgs_dir.parent /
                "labels"
            )

            if not labels_dir.exists():

                print(
                    f"\n[WARNING] Missing labels:"
                    f"\n{labels_dir}"
                )

                continue

            for image_path in sorted(
                imgs_dir.iterdir()
            ):

                if not image_path.is_file():
                    continue

                if (
                    image_path.suffix.lower()
                    not in IMAGE_EXTENSIONS
                ):
                    continue

                json_path = (
                    labels_dir /
                    f"{image_path.stem}.json"
                )

                if not json_path.exists():

                    missing_labels.append(
                        str(image_path)
                    )

                    continue

                pairs.append(
                    {
                        "image": image_path,
                        "json": json_path,
                        "case": case_name,
                    }
                )

    print(
        f"\nImage/JSON pairs found: "
        f"{len(pairs)}"
    )

    print(
        f"Missing JSON files: "
        f"{len(missing_labels)}"
    )

    return pairs


# ============================================================
# LOAD JSON
# ============================================================

def load_json(path):

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:

        return json.load(f)


# ============================================================
# IMAGE DIMENSIONS
# ============================================================

def get_actual_dimensions(image_path):

    with Image.open(image_path) as img:

        return img.size


# ============================================================
# ROTATION FUNCTIONS
# ============================================================

def rotate_box(
    box,
    width,
    height,
    rotation,
):
    """
    Convert an XYXY box from a rotated coordinate system
    back into the original image coordinate system.

    Input:
        box = [x1, y1, x2, y2]

    rotation:
        0
        90
        180
        270

    The formulas operate on all four corners, which is safer
    than trying to transform only x1/y1/x2/y2.
    """

    x1, y1, x2, y2 = box

    corners = [
        (x1, y1),
        (x2, y1),
        (x2, y2),
        (x1, y2),
    ]

    transformed = []

    for x, y in corners:

        if rotation == 0:

            nx = x
            ny = y

        elif rotation == 90:

            # 90 degrees clockwise
            nx = height - y
            ny = x

        elif rotation == 180:

            nx = width - x
            ny = height - y

        elif rotation == 270:

            # 90 degrees counter-clockwise
            nx = y
            ny = width - x

        else:

            raise ValueError(
                f"Unsupported rotation: {rotation}"
            )

        transformed.append(
            (nx, ny)
        )

    xs = [
        p[0]
        for p in transformed
    ]

    ys = [
        p[1]
        for p in transformed
    ]

    return [
        min(xs),
        min(ys),
        max(xs),
        max(ys),
    ]


# ============================================================
# NORMALIZE XYXY
# ============================================================

def normalize_xyxy(box):

    x1, y1, x2, y2 = box

    return [
        min(x1, x2),
        min(y1, y2),
        max(x1, x2),
        max(y1, y2),
    ]


# ============================================================
# VALIDATE BOX
# ============================================================

def validate_box(
    box,
    width,
    height,
):

    if box is None:
        return False

    x1, y1, x2, y2 = box

    if x2 <= x1:
        return False

    if y2 <= y1:
        return False

    if x1 < 0:
        return False

    if y1 < 0:
        return False

    if x2 > width:
        return False

    if y2 > height:
        return False

    return True


# ============================================================
# BOX QUALITY
# ============================================================

def box_quality(
    box,
    width,
    height,
):
    """
    Returns a score describing how plausible a box is.

    This is NOT used to invent annotations.
    It is only used when several rotations are possible.
    """

    if not validate_box(
        box,
        width,
        height,
    ):

        return -1

    x1, y1, x2, y2 = box

    box_width = x2 - x1
    box_height = y2 - y1

    area = (
        box_width *
        box_height
    )

    image_area = (
        width *
        height
    )

    area_ratio = (
        area /
        image_area
    )

    score = 100

    # Penalize boxes covering nearly the whole image
    if area_ratio > 0.5:
        score -= 50

    elif area_ratio > 0.25:
        score -= 25

    elif area_ratio > 0.10:
        score -= 10

    # Ball boxes generally shouldn't be extremely thin
    ratio = (
        max(box_width, box_height) /
        min(box_width, box_height)
    )

    if ratio > 10:
        score -= 20

    return score


# ============================================================
# FIND BEST ROTATION
# ============================================================

def determine_rotation(
    source_box,
    image_width,
    image_height,
):
    """
    Test 0/90/180/270.

    IMPORTANT:
    If more than one rotation is valid, we mark the result
    as AMBIGUOUS instead of silently guessing.
    """

    candidates = []

    for rotation in [
        0,
        90,
        180,
        270,
    ]:

        transformed = rotate_box(
            source_box,
            image_width,
            image_height,
            rotation,
        )

        transformed = normalize_xyxy(
            transformed
        )

        if validate_box(
            transformed,
            image_width,
            image_height,
        ):

            score = box_quality(
                transformed,
                image_width,
                image_height,
            )

            candidates.append(
                (
                    rotation,
                    transformed,
                    score,
                )
            )

    if len(candidates) == 0:

        return None, "INVALID", []

    if len(candidates) == 1:

        rotation, box, score = (
            candidates[0]
        )

        return (
            rotation,
            "UNIQUE",
            candidates,
        )

    # Multiple valid rotations
    #
    # DO NOT blindly choose one.
    #

    candidates.sort(
        key=lambda x: x[2],
        reverse=True
    )

    best = candidates[0]

    second = candidates[1]

    if best[2] > second[2]:

        # Still flag because this is heuristic
        return (
            best[0],
            "HEURISTIC",
            candidates,
        )

    return (
        None,
        "AMBIGUOUS",
        candidates,
    )


# ============================================================
# COCO CONVERSION
# ============================================================

def xyxy_to_coco(box):

    x1, y1, x2, y2 = box

    width = x2 - x1
    height = y2 - y1

    return [
        float(x1),
        float(y1),
        float(width),
        float(height),
    ]


# ============================================================
# PROCESS ONE IMAGE
# ============================================================

def process_image(
    item,
    diagnostic_rows,
):
    """
    Process one image + JSON.

    Returns:

        image_width
        image_height
        valid_boxes
    """

    image_path = item["image"]

    json_path = item["json"]

    annotation = load_json(
        json_path
    )

    actual_width, actual_height = (
        get_actual_dimensions(
            image_path
        )
    )

    json_dimensions = annotation.get(
        "dimensions"
    )

    # --------------------------------------------------------
    # Record JSON dimensions
    # --------------------------------------------------------

    if (
        isinstance(json_dimensions, list)
        and len(json_dimensions) == 2
    ):

        json_width = json_dimensions[0]
        json_height = json_dimensions[1]

    else:

        json_width = None
        json_height = None

    # --------------------------------------------------------
    # Dimension mismatch
    # --------------------------------------------------------

    dimension_match = (
        json_width == actual_width
        and
        json_height == actual_height
    )

    # --------------------------------------------------------
    # Balls
    # --------------------------------------------------------

    data = annotation.get(
        "data",
        {}
    )

    balls = data.get(
        "ball",
        []
    )

    valid_boxes = []

    for ball_index, ball in enumerate(
        balls
    ):

        try:

            source_box = (
                ball[
                    "entire"
                ][
                    "rect"
                ]
            )

        except (
            KeyError,
            TypeError,
        ):

            diagnostic_rows.append(
                {
                    "image": str(image_path),
                    "json": str(json_path),
                    "ball_index": ball_index,
                    "source_rect": "",
                    "actual_width": actual_width,
                    "actual_height": actual_height,
                    "json_width": json_width,
                    "json_height": json_height,
                    "dimension_match": dimension_match,
                    "rotation": "",
                    "status": "MISSING_RECT",
                    "coco_bbox": "",
                }
            )

            continue

        # ----------------------------------------------------
        # Check source rect
        # ----------------------------------------------------

        if (
            not isinstance(source_box, list)
            or len(source_box) != 4
        ):

            diagnostic_rows.append(
                {
                    "image": str(image_path),
                    "json": str(json_path),
                    "ball_index": ball_index,
                    "source_rect": str(source_box),
                    "actual_width": actual_width,
                    "actual_height": actual_height,
                    "json_width": json_width,
                    "json_height": json_height,
                    "dimension_match": dimension_match,
                    "rotation": "",
                    "status": "BAD_RECT",
                    "coco_bbox": "",
                }
            )

            continue

        source_box = [
            float(v)
            for v in source_box
        ]

        # ----------------------------------------------------
        # Forced rotation
        # ----------------------------------------------------

        if ROTATION != "auto":

            rotation = int(
                ROTATION
            )

            box = rotate_box(
                source_box,
                actual_width,
                actual_height,
                rotation,
            )

            box = normalize_xyxy(
                box
            )

            status = (
                "VALID"
                if validate_box(
                    box,
                    actual_width,
                    actual_height,
                )
                else "INVALID"
            )

        # ----------------------------------------------------
        # Automatic rotation diagnosis
        # ----------------------------------------------------

        else:

            rotation, status, candidates = (
                determine_rotation(
                    source_box,
                    actual_width,
                    actual_height,
                )
            )

            if rotation is None:

                box = None

            else:

                box = rotate_box(
                    source_box,
                    actual_width,
                    actual_height,
                    rotation,
                )

                box = normalize_xyxy(
                    box
                )

        # ----------------------------------------------------
        # Convert valid box
        # ----------------------------------------------------

        coco_bbox = ""

        if box is not None and validate_box(
            box,
            actual_width,
            actual_height,
        ):

            coco_bbox = xyxy_to_coco(
                box
            )

            valid_boxes.append(
                coco_bbox
            )

        # ----------------------------------------------------
        # Diagnostic
        # ----------------------------------------------------

        diagnostic_rows.append(
            {
                "image": str(image_path),
                "json": str(json_path),
                "ball_index": ball_index,
                "source_rect": str(source_box),
                "actual_width": actual_width,
                "actual_height": actual_height,
                "json_width": json_width,
                "json_height": json_height,
                "dimension_match": dimension_match,
                "rotation": (
                    rotation
                    if rotation is not None
                    else ""
                ),
                "status": status,
                "coco_bbox": str(coco_bbox),
            }
        )

    return (
        actual_width,
        actual_height,
        valid_boxes,
    )


# ============================================================
# UNIQUE OUTPUT FILENAME
# ============================================================

def unique_filename(item):

    image_path = item["image"]

    case_name = item["case"]

    # Example:
    #
    # basic_train_caseA/
    # basketball/
    # imgs/
    #
    # category = basketball

    category = (
        image_path.parent.parent.name
    )

    return (
        f"{case_name}_"
        f"{category}_"
        f"{image_path.name}"
    )


# ============================================================
# CREATE COCO SPLIT
# ============================================================

def create_split(
    items,
    split_name,
    diagnostic_rows,
):

    image_dir = (
        OUTPUT_ROOT /
        "images" /
        split_name
    )

    image_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    coco = {

        "info": {
            "description": (
                "Ball Detection Dataset"
            ),
            "version": "1.0",
        },

        "licenses": [],

        "images": [],

        "annotations": [],

        "categories": CATEGORIES,
    }

    image_id = 1

    annotation_id = 1

    copied_names = set()

    skipped_images = 0

    for item in items:

        image_path = item["image"]

        try:

            width, height, boxes = (
                process_image(
                    item,
                    diagnostic_rows,
                )
            )

        except Exception as e:

            print(
                f"\n[ERROR]"
                f"\n{image_path}"
                f"\n{e}"
            )

            skipped_images += 1

            continue

        # ----------------------------------------------------
        # If no valid boxes, skip image
        # ----------------------------------------------------

        if len(boxes) == 0:

            skipped_images += 1

            continue

        # ----------------------------------------------------
        # Filename
        # ----------------------------------------------------

        filename = unique_filename(
            item
        )

        if filename in copied_names:

            print(
                f"\n[ERROR] Duplicate filename:"
                f"\n{filename}"
            )

            skipped_images += 1

            continue

        copied_names.add(
            filename
        )

        destination = (
            image_dir /
            filename
        )

        shutil.copy2(
            image_path,
            destination
        )

        # ----------------------------------------------------
        # COCO image
        # ----------------------------------------------------

        coco["images"].append(
            {
                "id": image_id,
                "file_name": filename,
                "width": width,
                "height": height,
            }
        )

        # ----------------------------------------------------
        # COCO annotations
        # ----------------------------------------------------

        for bbox in boxes:

            x, y, w, h = bbox

            coco["annotations"].append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": 1,
                    "bbox": [
                        x,
                        y,
                        w,
                        h,
                    ],
                    "area": w * h,
                    "iscrowd": 0,
                }
            )

            annotation_id += 1

        image_id += 1

    # --------------------------------------------------------
    # Save JSON
    # --------------------------------------------------------

    annotation_dir = (
        OUTPUT_ROOT /
        "annotations"
    )

    annotation_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    json_path = (
        annotation_dir /
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
        f"  Images      : "
        f"{len(coco['images'])}"
    )

    print(
        f"  Annotations : "
        f"{len(coco['annotations'])}"
    )

    print(
        f"  Skipped     : "
        f"{skipped_images}"
    )

    print(
        f"  JSON        : "
        f"{json_path}"
    )

    return coco


# ============================================================
# SAVE DIAGNOSTICS
# ============================================================

def save_diagnostics(rows):

    path = (
        OUTPUT_ROOT /
        "bbox_diagnostics.csv"
    )

    fields = [
        "image",
        "json",
        "ball_index",
        "source_rect",
        "actual_width",
        "actual_height",
        "json_width",
        "json_height",
        "dimension_match",
        "rotation",
        "status",
        "coco_bbox",
    ]

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fields
        )

        writer.writeheader()

        writer.writerows(rows)

    print(
        f"\nDiagnostics saved:"
        f"\n  {path}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "\n" +
        "=" * 70
    )

    print(
        "ROBUST COCO DATASET CONVERTER"
    )

    print(
        "=" * 70
    )

    print(
        f"\nSource:"
        f"\n  {SOURCE_ROOT}"
    )

    print(
        f"\nOutput:"
        f"\n  {OUTPUT_ROOT}"
    )

    print(
        "\nIncluded:"
    )

    for case in SELECTED_CASES:

        print(
            f"  + {case}"
        )

    print(
        "\nExcluded:"
    )

    for folder in sorted(
        EXCLUDED_FOLDERS
    ):

        print(
            f"  - {folder}"
        )

    print(
        "\nSource rect interpretation:"
    )

    print(
        "  [x1, y1, x2, y2]"
    )

    print(
        "\nCOCO output:"
    )

    print(
        "  [x, y, width, height]"
    )

    print(
        "\nRotation mode:"
    )

    print(
        f"  {ROTATION}"
    )

    # --------------------------------------------------------
    # Validate split
    # --------------------------------------------------------

    total_ratio = (
        TRAIN_RATIO +
        VAL_RATIO +
        TEST_RATIO
    )

    if abs(
        total_ratio - 1.0
    ) > 0.0001:

        raise ValueError(
            "Split ratios must sum to 1.0"
        )

    # --------------------------------------------------------
    # Find pairs
    # --------------------------------------------------------

    pairs = find_pairs()

    if not pairs:

        print(
            "\nNo images found."
        )

        return

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

    print(
        "\n" +
        "=" * 70
    )

    print(
        "SPLIT"
    )

    print(
        "=" * 70
    )

    print(
        f"Total : {total}"
    )

    print(
        f"Train : {len(train_items)} "
        f"({len(train_items)/total*100:.2f}%)"
    )

    print(
        f"Val   : {len(val_items)} "
        f"({len(val_items)/total*100:.2f}%)"
    )

    print(
        f"Test  : {len(test_items)} "
        f"({len(test_items)/total*100:.2f}%)"
    )

    # --------------------------------------------------------
    # Remove old output
    # --------------------------------------------------------

    if OUTPUT_ROOT.exists():

        print(
            f"\nOutput already exists:"
            f"\n{OUTPUT_ROOT}"
        )

        answer = input(
            "\nDelete it and recreate? "
            "[y/N]: "
        ).strip().lower()

        if answer != "y":

            print(
                "\nStopped."
            )

            return

        shutil.rmtree(
            OUTPUT_ROOT
        )

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Diagnostics
    # --------------------------------------------------------

    diagnostic_rows = []

    # --------------------------------------------------------
    # Create splits
    # --------------------------------------------------------

    create_split(
        train_items,
        "train",
        diagnostic_rows,
    )

    create_split(
        val_items,
        "val",
        diagnostic_rows,
    )

    create_split(
        test_items,
        "test",
        diagnostic_rows,
    )

    # --------------------------------------------------------
    # Diagnostics CSV
    # --------------------------------------------------------

    save_diagnostics(
        diagnostic_rows
    )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    status_counts = Counter(
        row["status"]
        for row in diagnostic_rows
    )

    print(
        "\n" +
        "=" * 70
    )

    print(
        "ANNOTATION DIAGNOSTICS"
    )

    print(
        "=" * 70
    )

    for status, count in (
        status_counts.items()
    ):

        print(
            f"{status:15s}: {count}"
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
{OUTPUT_ROOT}/
│
├── images/
│   ├── train/
│   ├── val/
│   └── test/
│
├── annotations/
│   ├── instances_train.json
│   ├── instances_val.json
│   └── instances_test.json
│
└── bbox_diagnostics.csv
"""
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()
