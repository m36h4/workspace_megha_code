import json
import random
import shutil
import csv
from pathlib import Path
from collections import Counter

from PIL import Image, ImageOps


# ============================================================
# CONFIG
# ============================================================

SOURCE_ROOT = Path("/home/eng_megha/balldataset")

OUTPUT_ROOT = Path("/home/eng_megha/balldataset_coco")


# ONLY these four
SELECTED_CASES = [
    "AI_train_caseB",
    "basic_train_caseA",
    "cc0_train_caseB",
    "match_train_caseA",
]


# Explicitly ignored
EXCLUDED_FOLDERS = {
    "aug",
    "other",
    "volleyball",
}


TRAIN_RATIO = 0.70
VAL_RATIO = 0.20
TEST_RATIO = 0.10

RANDOM_SEED = 42


# ============================================================
# SOURCE RECT FORMAT
# ============================================================
#
# IMPORTANT:
#
# We are treating your source annotation as:
#
#       [x1, y1, x2, y2]
#
# NOT:
#
#       [x, y, width, height]
#
# ============================================================


SOURCE_FORMAT = "xyxy"


# ============================================================
# IMAGE ORIENTATION
# ============================================================
#
# "exif" means:
#
#   - read the image EXIF orientation
#   - physically correct the orientation
#   - transform the bounding box accordingly
#
# This is much safer than guessing 90/180/270 from a bbox.
#
# ============================================================

USE_EXIF_ORIENTATION = True


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
# EXCLUDED PATH
# ============================================================

def is_excluded(path):

    for part in path.parts:

        if part.lower() in EXCLUDED_FOLDERS:
            return True

    return False


# ============================================================
# FIND IMAGE / JSON PAIRS
# ============================================================

def find_pairs():

    pairs = []

    for case_name in SELECTED_CASES:

        case_root = SOURCE_ROOT / case_name

        if not case_root.exists():

            print(
                f"[WARNING] Missing:"
                f" {case_root}"
            )

            continue

        print(
            f"Scanning: {case_name}"
        )

        for imgs_dir in case_root.rglob("imgs"):

            if not imgs_dir.is_dir():
                continue

            if is_excluded(imgs_dir):
                continue

            labels_dir = (
                imgs_dir.parent /
                "labels"
            )

            if not labels_dir.exists():

                print(
                    f"[WARNING] No labels:"
                    f" {labels_dir}"
                )

                continue

            for image_path in imgs_dir.iterdir():

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

                    print(
                        f"[WARNING] Missing JSON:"
                        f" {image_path}"
                    )

                    continue

                pairs.append(
                    {
                        "image": image_path,
                        "json": json_path,
                        "case": case_name,
                    }
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
# GET IMAGE ORIENTATION
# ============================================================

def get_exif_orientation(image):

    try:

        exif = image.getexif()

        # EXIF orientation tag = 274

        return exif.get(
            274,
            1
        )

    except Exception:

        return 1


# ============================================================
# CORRECT IMAGE ORIENTATION
# ============================================================

def correct_image_orientation(image):

    """
    PIL's ImageOps.exif_transpose physically rotates/flips
    the image according to EXIF orientation.

    The returned image is therefore in normal visual
    orientation.
    """

    if not USE_EXIF_ORIENTATION:

        return image

    return ImageOps.exif_transpose(
        image
    )


# ============================================================
# TRANSFORM XYXY ACCORDING TO EXIF
# ============================================================

def transform_bbox_exif(
    bbox,
    width,
    height,
    orientation,
):
    """
    Transform a bbox according to EXIF orientation.

    Input bbox:
        [x1, y1, x2, y2]

    Coordinates are assumed to belong to the raw image.

    Output bbox belongs to the EXIF-corrected image.
    """

    x1, y1, x2, y2 = bbox

    corners = [
        (x1, y1),
        (x2, y1),
        (x2, y2),
        (x1, y2),
    ]

    transformed = []

    for x, y in corners:

        # ----------------------------------------------------
        # EXIF 1
        # Normal
        # ----------------------------------------------------

        if orientation == 1:

            nx = x
            ny = y

        # ----------------------------------------------------
        # EXIF 2
        # Mirror horizontal
        # ----------------------------------------------------

        elif orientation == 2:

            nx = width - x
            ny = y

        # ----------------------------------------------------
        # EXIF 3
        # Rotate 180
        # ----------------------------------------------------

        elif orientation == 3:

            nx = width - x
            ny = height - y

        # ----------------------------------------------------
        # EXIF 4
        # Mirror vertical
        # ----------------------------------------------------

        elif orientation == 4:

            nx = x
            ny = height - y

        # ----------------------------------------------------
        # EXIF 5
        # Mirror horizontal + rotate 270 CW
        # ----------------------------------------------------

        elif orientation == 5:

            nx = y
            ny = x

        # ----------------------------------------------------
        # EXIF 6
        # Rotate 90 CW
        # ----------------------------------------------------

        elif orientation == 6:

            nx = height - y
            ny = x

        # ----------------------------------------------------
        # EXIF 7
        # Mirror horizontal + rotate 90 CW
        # ----------------------------------------------------

        elif orientation == 7:

            nx = height - y
            ny = width - x

        # ----------------------------------------------------
        # EXIF 8
        # Rotate 270 CW
        # ----------------------------------------------------

        elif orientation == 8:

            nx = y
            ny = width - x

        else:

            nx = x
            ny = y

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
# NORMALIZE BOX
# ============================================================

def normalize_xyxy(bbox):

    x1, y1, x2, y2 = bbox

    return [
        min(x1, x2),
        min(y1, y2),
        max(x1, x2),
        max(y1, y2),
    ]


# ============================================================
# CLIP BOX TO IMAGE
# ============================================================

def clip_bbox(
    bbox,
    width,
    height,
):
    """
    Keep a box inside image boundaries.

    This is important because some annotations can touch
    or slightly exceed image boundaries.
    """

    x1, y1, x2, y2 = bbox

    x1 = max(
        0,
        min(x1, width)
    )

    y1 = max(
        0,
        min(y1, height)
    )

    x2 = max(
        0,
        min(x2, width)
    )

    y2 = max(
        0,
        min(y2, height)
    )

    return [
        x1,
        y1,
        x2,
        y2,
    ]


# ============================================================
# XYXY -> COCO
# ============================================================

def xyxy_to_coco(bbox):

    x1, y1, x2, y2 = bbox

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
    diagnostics,
):

    image_path = item["image"]
    json_path = item["json"]

    annotation = load_json(
        json_path
    )

    # --------------------------------------------------------
    # Open original image
    # --------------------------------------------------------

    with Image.open(
        image_path
    ) as original:

        raw_width, raw_height = (
            original.size
        )

        orientation = (
            get_exif_orientation(
                original
            )
        )

        # Correct visual orientation
        corrected = (
            correct_image_orientation(
                original
            )
        )

        corrected_width, corrected_height = (
            corrected.size
        )

    # --------------------------------------------------------
    # JSON dimensions
    # --------------------------------------------------------

    dimensions = annotation.get(
        "dimensions"
    )

    if (
        isinstance(dimensions, list)
        and len(dimensions) == 2
    ):

        json_dim_1 = int(
            dimensions[0]
        )

        json_dim_2 = int(
            dimensions[1]
        )

    else:

        json_dim_1 = None
        json_dim_2 = None

    # --------------------------------------------------------
    # Balls
    # --------------------------------------------------------

    balls = (
        annotation
        .get("data", {})
        .get("ball", [])
    )

    coco_boxes = []

    for ball_index, ball in enumerate(
        balls
    ):

        try:

            rect = (
                ball
                ["entire"]
                ["rect"]
            )

        except (
            KeyError,
            TypeError,
        ):

            diagnostics.append(
                {
                    "image": str(image_path),
                    "json": str(json_path),
                    "ball_index": ball_index,
                    "rect": "",
                    "raw_size":
                        f"{raw_width}x{raw_height}",
                    "corrected_size":
                        f"{corrected_width}x{corrected_height}",
                    "exif_orientation":
                        orientation,
                    "status":
                        "MISSING_RECT",
                }
            )

            continue

        if (
            not isinstance(rect, list)
            or len(rect) != 4
        ):

            diagnostics.append(
                {
                    "image": str(image_path),
                    "json": str(json_path),
                    "ball_index": ball_index,
                    "rect": str(rect),
                    "raw_size":
                        f"{raw_width}x{raw_height}",
                    "corrected_size":
                        f"{corrected_width}x{corrected_height}",
                    "exif_orientation":
                        orientation,
                    "status":
                        "BAD_RECT",
                }
            )

            continue

        rect = [
            float(v)
            for v in rect
        ]

        # ----------------------------------------------------
        # SOURCE XYXY
        # ----------------------------------------------------

        source_bbox = normalize_xyxy(
            rect
        )

        # ----------------------------------------------------
        # Transform EXIF orientation
        # ----------------------------------------------------

        if (
            USE_EXIF_ORIENTATION
            and orientation != 1
        ):

            bbox = transform_bbox_exif(
                source_bbox,
                raw_width,
                raw_height,
                orientation,
            )

        else:

            bbox = source_bbox

        # ----------------------------------------------------
        # Normalize again
        # ----------------------------------------------------

        bbox = normalize_xyxy(
            bbox
        )

        # ----------------------------------------------------
        # Clip
        # ----------------------------------------------------

        bbox = clip_bbox(
            bbox,
            corrected_width,
            corrected_height,
        )

        # ----------------------------------------------------
        # Check size
        # ----------------------------------------------------

        x1, y1, x2, y2 = bbox

        box_width = (
            x2 - x1
        )

        box_height = (
            y2 - y1
        )

        # ----------------------------------------------------
        # Reject only genuinely zero/negative boxes
        # ----------------------------------------------------

        if (
            box_width <= 0
            or
            box_height <= 0
        ):

            diagnostics.append(
                {
                    "image": str(image_path),
                    "json": str(json_path),
                    "ball_index": ball_index,
                    "rect": str(rect),
                    "raw_size":
                        f"{raw_width}x{raw_height}",
                    "corrected_size":
                        f"{corrected_width}x{corrected_height}",
                    "exif_orientation":
                        orientation,
                    "status":
                        "INVALID_AFTER_TRANSFORM",
                }
            )

            continue

        # ----------------------------------------------------
        # Convert to COCO
        # ----------------------------------------------------

        coco_bbox = xyxy_to_coco(
            bbox
        )

        coco_boxes.append(
            coco_bbox
        )

        diagnostics.append(
            {
                "image": str(image_path),
                "json": str(json_path),
                "ball_index": ball_index,
                "rect": str(rect),
                "raw_size":
                    f"{raw_width}x{raw_height}",
                "corrected_size":
                    f"{corrected_width}x{corrected_height}",
                "exif_orientation":
                    orientation,
                "status":
                    "VALID",
            }
        )

    return (
        corrected_width,
        corrected_height,
        orientation,
        coco_boxes,
    )


# ============================================================
# OUTPUT FILENAME
# ============================================================

def make_filename(item):

    image_path = item["image"]

    case_name = item["case"]

    category = (
        image_path
        .parent
        .parent
        .name
    )

    return (
        f"{case_name}_"
        f"{category}_"
        f"{image_path.name}"
    )


# ============================================================
# CREATE SPLIT
# ============================================================

def create_split(
    items,
    split_name,
    diagnostics,
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
            "description":
                "Ball Detection Dataset",
            "version":
                "1.0",
        },

        "licenses": [],

        "images": [],

        "annotations": [],

        "categories":
            CATEGORIES,
    }

    image_id = 1
    annotation_id = 1

    skipped = 0

    for item in items:

        try:

            (
                width,
                height,
                orientation,
                boxes,
            ) = process_image(
                item,
                diagnostics,
            )

        except Exception as e:

            print(
                f"\n[ERROR]"
                f"\n{item['image']}"
                f"\n{e}"
            )

            skipped += 1

            continue

        # ----------------------------------------------------
        # IMPORTANT:
        #
        # If an image has zero annotations, we keep it only
        # if you want negative/background images.
        #
        # For ball detection, we keep it here.
        # ----------------------------------------------------

        filename = make_filename(
            item
        )

        destination = (
            image_dir /
            filename
        )

        # ----------------------------------------------------
        # Copy image
        # ----------------------------------------------------

        try:

            with Image.open(
                item["image"]
            ) as img:

                if USE_EXIF_ORIENTATION:

                    img = ImageOps.exif_transpose(
                        img
                    )

                # Save physically corrected image
                #
                # This removes the dependency on EXIF
                # orientation during training.

                img.save(
                    destination,
                    quality=95
                )

        except Exception as e:

            print(
                f"\n[ERROR] Could not save:"
                f"\n{item['image']}"
                f"\n{e}"
            )

            skipped += 1

            continue

        # ----------------------------------------------------
        # COCO IMAGE
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
        # COCO BOXES
        # ----------------------------------------------------

        for bbox in boxes:

            x, y, w, h = bbox

            coco["annotations"].append(
                {
                    "id":
                        annotation_id,

                    "image_id":
                        image_id,

                    "category_id":
                        1,

                    "bbox":
                        [
                            x,
                            y,
                            w,
                            h,
                        ],

                    "area":
                        w * h,

                    "iscrowd":
                        0,
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
        f"{skipped}"
    )

    print(
        f"  JSON        : "
        f"{json_path}"
    )


# ============================================================
# SAVE DIAGNOSTICS
# ============================================================

def save_diagnostics(rows):

    output = (
        OUTPUT_ROOT /
        "bbox_diagnostics.csv"
    )

    fields = [
        "image",
        "json",
        "ball_index",
        "rect",
        "raw_size",
        "corrected_size",
        "exif_orientation",
        "status",
    ]

    with open(
        output,
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
        f"\nDiagnostics:"
        f"\n{output}"
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
        "COCO DATASET CONVERTER"
    )

    print(
        "=" * 70
    )

    print(
        "\nIMPORTANT:"
    )

    print(
        "Source rect = [x1, y1, x2, y2]"
    )

    print(
        "COCO bbox  = [x, y, width, height]"
    )

    print(
        "EXIF orientation correction = ON"
    )

    # --------------------------------------------------------
    # Find all data
    # --------------------------------------------------------

    pairs = find_pairs()

    print(
        f"\nImage/JSON pairs found:"
        f" {len(pairs)}"
    )

    if not pairs:
        return

    # --------------------------------------------------------
    # Random split
    # --------------------------------------------------------

    random.seed(
        RANDOM_SEED
    )

    random.shuffle(
        pairs
    )

    total = len(pairs)

    train_count = int(
        total * TRAIN_RATIO
    )

    val_count = int(
        total * VAL_RATIO
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
        "\nSPLIT:"
    )

    print(
        f"Train: {len(train_items)}"
    )

    print(
        f"Val  : {len(val_items)}"
    )

    print(
        f"Test : {len(test_items)}"
    )

    # --------------------------------------------------------
    # Delete old output
    # --------------------------------------------------------

    if OUTPUT_ROOT.exists():

        print(
            f"\nOutput already exists:"
            f"\n{OUTPUT_ROOT}"
        )

        answer = input(
            "Delete it? [y/N]: "
        ).strip().lower()

        if answer != "y":

            print(
                "Stopped."
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

    diagnostics = []

    # --------------------------------------------------------
    # Create datasets
    # --------------------------------------------------------

    create_split(
        train_items,
        "train",
        diagnostics,
    )

    create_split(
        val_items,
        "val",
        diagnostics,
    )

    create_split(
        test_items,
        "test",
        diagnostics,
    )

    # --------------------------------------------------------
    # Save diagnostics
    # --------------------------------------------------------

    save_diagnostics(
        diagnostics
    )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    counts = Counter(
        row["status"]
        for row in diagnostics
    )

    print(
        "\n" +
        "=" * 70
    )

    print(
        "ANNOTATION SUMMARY"
    )

    print(
        "=" * 70
    )

    for status, count in counts.items():

        print(
            f"{status:30s}: {count}"
        )

    # --------------------------------------------------------
    # Final
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
