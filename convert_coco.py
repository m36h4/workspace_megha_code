import json
import random
import shutil
from pathlib import Path
from PIL import Image


# ============================================================
# CONFIGURATION
# ============================================================

# CHANGE THIS
SOURCE_ROOT = Path("/home/eng_megha/balldataset")

# Main output directory
OUTPUT_ROOT = Path("/home/eng_megha/balldataset_processed")


# ============================================================
# DATASET CASES
# ============================================================

# These cases participate in 70/20/10 split
SPLIT_CASES = [
    "AI_train_caseB",
    "basic_train_caseA",
    "cc0_train_caseB",
    "match_train_caseA",
]

# This case is ONLY converted to COCO.
# It is NOT used for train/val/test.
EVAL_CASE = "basic_eval_caseA"


# Cases whose original annotations are XYXY
XYXY_CASES = {
    "AI_train_caseB",
    "basic_train_caseA",
    "basic_eval_caseA",
}

# These are already XYWH
XYWH_CASES = {
    "cc0_train_caseB",
    "match_train_caseA",
}


# Folders that must never be used
EXCLUDED_FOLDERS = {
    "aug",
    "other",
    "volleyball",
}


# ============================================================
# SPLIT
# ============================================================

TRAIN_RATIO = 0.70
VAL_RATIO = 0.20
TEST_RATIO = 0.10

RANDOM_SEED = 42


# ============================================================
# IMAGE EXTENSIONS
# ============================================================

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
# CHECK CONFIGURATION
# ============================================================

assert abs(
    TRAIN_RATIO + VAL_RATIO + TEST_RATIO - 1.0
) < 1e-6


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def is_excluded(path):
    """
    Returns True if the path contains one of the folders
    that should be ignored.
    """

    return any(
        part.lower() in EXCLUDED_FOLDERS
        for part in path.parts
    )


def find_imgs_directories(case_root):
    """
    Find every imgs directory under a case.
    """

    return [
        p
        for p in case_root.rglob("imgs")
        if p.is_dir() and not is_excluded(p)
    ]


def convert_xyxy_to_xywh(rect):
    """
    Convert:

        [x1, y1, x2, y2]

    to:

        [x, y, width, height]
    """

    if not isinstance(rect, list) or len(rect) != 4:
        raise ValueError(
            f"Invalid rect: {rect}"
        )

    x1, y1, x2, y2 = rect

    width = x2 - x1
    height = y2 - y1

    if width < 0 or height < 0:
        raise ValueError(
            f"Invalid XYXY box: {rect}"
        )

    return [
        x1,
        y1,
        width,
        height,
    ]


def normalize_annotation(
    annotation,
    annotation_format
):
    """
    Convert an annotation to the common XYWH format.

    Only the 'rect' field is changed.

    Everything else is preserved.
    """

    annotation = json.loads(
        json.dumps(annotation)
    )

    data = annotation.get("data", {})

    balls = data.get("ball", [])

    for ball in balls:

        entire = ball.get("entire", {})

        rect = entire.get("rect")

        if rect is None:
            continue

        if annotation_format == "xyxy":

            entire["rect"] = convert_xyxy_to_xywh(rect)

        elif annotation_format == "xywh":

            # Already correct.
            entire["rect"] = rect

        else:
            raise ValueError(
                f"Unknown annotation format: "
                f"{annotation_format}"
            )

    return annotation


def get_annotation_format(case_name):
    """
    Return the known format for each case.
    """

    if case_name in XYXY_CASES:
        return "xyxy"

    if case_name in XYWH_CASES:
        return "xywh"

    raise ValueError(
        f"Annotation format not defined for: {case_name}"
    )


def get_unique_image_name(
    case_name,
    imgs_dir,
    image_name
):
    """
    Prevent duplicate image names.

    Example:

        AI_train_caseB_basketball_image1.jpg

    """

    # Get the sport/folder immediately above imgs
    relative_parent = imgs_dir.parent.name

    return (
        f"{case_name}_"
        f"{relative_parent}_"
        f"{image_name}"
    )


# ============================================================
# CREATE UPDATED LABELS
# ============================================================

def create_updated_labels(case_names):
    """
    Create a clean updated_labels directory.

    XYXY -> XYWH

    XYWH -> copied unchanged

    Original labels are NEVER modified.
    """

    print("\n")
    print("=" * 70)
    print("STEP 1: CREATING UPDATED LABELS")
    print("=" * 70)

    updated_root = OUTPUT_ROOT / "updated_labels"

    updated_root.mkdir(
        parents=True,
        exist_ok=True
    )

    all_pairs = []

    for case_name in case_names:

        case_root = SOURCE_ROOT / case_name

        if not case_root.exists():

            print(
                f"[WARNING] Case not found: "
                f"{case_root}"
            )

            continue

        annotation_format = get_annotation_format(
            case_name
        )

        print(
            f"\n{case_name} "
            f"[{annotation_format.upper()}]"
        )

        imgs_dirs = find_imgs_directories(
            case_root
        )

        for imgs_dir in imgs_dirs:

            labels_dir = (
                imgs_dir.parent / "labels"
            )

            if not labels_dir.exists():

                print(
                    f"  [WARNING] Labels not found: "
                    f"{labels_dir}"
                )

                continue

            # ------------------------------------------------
            # Preserve directory structure
            # ------------------------------------------------

            relative_case_path = (
                imgs_dir.parent.relative_to(case_root)
            )

            output_labels_dir = (
                updated_root /
                case_name /
                relative_case_path /
                "labels"
            )

            output_labels_dir.mkdir(
                parents=True,
                exist_ok=True
            )

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

                label_path = (
                    labels_dir /
                    f"{image_path.stem}.json"
                )

                if not label_path.exists():

                    print(
                        f"  [MISSING LABEL] "
                        f"{image_path}"
                    )

                    continue

                # ------------------------------------------------
                # Read original annotation
                # ------------------------------------------------

                try:

                    with open(
                        label_path,
                        "r",
                        encoding="utf-8"
                    ) as f:

                        annotation = json.load(f)

                except Exception as e:

                    print(
                        f"  [ERROR] "
                        f"{label_path}: {e}"
                    )

                    continue

                # ------------------------------------------------
                # Convert XYXY -> XYWH
                # ------------------------------------------------

                try:

                    updated_annotation = (
                        normalize_annotation(
                            annotation,
                            annotation_format
                        )
                    )

                except Exception as e:

                    print(
                        f"  [ERROR] "
                        f"{label_path}: {e}"
                    )

                    continue

                # ------------------------------------------------
                # Save updated label
                # ------------------------------------------------

                output_label_path = (
                    output_labels_dir /
                    label_path.name
                )

                with open(
                    output_label_path,
                    "w",
                    encoding="utf-8"
                ) as f:

                    json.dump(
                        updated_annotation,
                        f,
                        indent=2
                    )

                all_pairs.append({
                    "case": case_name,
                    "image": image_path,
                    "updated_label": output_label_path,
                    "imgs_dir": imgs_dir,
                })

    print(
        f"\nTotal normalized image-label pairs: "
        f"{len(all_pairs)}"
    )

    print(
        f"Updated labels saved to:\n"
        f"{updated_root}"
    )

    return all_pairs


# ============================================================
# CREATE COCO DATASET
# ============================================================

def create_coco(
    pairs,
    split_name,
    output_root,
):
    """
    Convert normalized XYWH annotations into COCO.

    The input labels MUST already be XYWH.
    """

    print("\n")
    print("=" * 70)
    print(f"COCO CONVERSION: {split_name.upper()}")
    print("=" * 70)

    images_output_dir = (
        output_root /
        "images" /
        split_name
    )

    annotations_output_dir = (
        output_root /
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

    for image_id, item in enumerate(
        pairs,
        start=1
    ):

        image_path = item["image"]
        label_path = item["updated_label"]
        case_name = item["case"]
        imgs_dir = item["imgs_dir"]

        # ----------------------------------------------------
        # Read annotation
        # ----------------------------------------------------

        try:

            with open(
                label_path,
                "r",
                encoding="utf-8"
            ) as f:

                annotation_data = json.load(f)

        except Exception as e:

            print(
                f"[ERROR] Reading "
                f"{label_path}: {e}"
            )

            continue

        # ----------------------------------------------------
        # Dimensions
        # ----------------------------------------------------

        dimensions = (
            annotation_data.get("dimensions")
        )

        if (
            dimensions is not None
            and len(dimensions) == 2
        ):

            # Your format:
            # [height, width]

            height = int(dimensions[0])
            width = int(dimensions[1])

        else:

            # Fallback to actual image
            with Image.open(image_path) as im:

                width, height = im.size

        # ----------------------------------------------------
        # Unique image filename
        # ----------------------------------------------------

        unique_name = get_unique_image_name(
            case_name,
            imgs_dir,
            image_path.name
        )

        destination = (
            images_output_dir /
            unique_name
        )

        shutil.copy2(
            image_path,
            destination
        )

        # ----------------------------------------------------
        # COCO image
        # ----------------------------------------------------

        coco["images"].append({
            "id": image_id,
            "file_name": unique_name,
            "width": width,
            "height": height,
        })

        # ----------------------------------------------------
        # Ball annotations
        # ----------------------------------------------------

        data = annotation_data.get(
            "data",
            {}
        )

        balls = data.get(
            "ball",
            []
        )

        for ball in balls:

            entire = ball.get(
                "entire",
                {}
            )

            rect = entire.get(
                "rect"
            )

            if rect is None:
                continue

            if len(rect) != 4:
                print(
                    f"[WARNING] Bad bbox: "
                    f"{rect}"
                )
                continue

            # IMPORTANT:
            # At this point rect is already XYWH.

            x, y, bbox_width, bbox_height = (
                rect
            )

            # ------------------------------------------------
            # Validate bbox
            # ------------------------------------------------

            if bbox_width <= 0 or bbox_height <= 0:

                print(
                    f"[WARNING] Invalid bbox "
                    f"in {label_path}: {rect}"
                )

                continue

            # ------------------------------------------------
            # COCO annotation
            # ------------------------------------------------

            coco["annotations"].append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": 1,
                "bbox": [
                    float(x),
                    float(y),
                    float(bbox_width),
                    float(bbox_height),
                ],
                "area": float(
                    bbox_width * bbox_height
                ),
                "iscrowd": 0,
            })

            annotation_id += 1

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
    ) as f:

        json.dump(
            coco,
            f,
            indent=2
        )

    print(
        f"\nImages      : "
        f"{len(coco['images'])}"
    )

    print(
        f"Annotations : "
        f"{len(coco['annotations'])}"
    )

    print(
        f"Saved       : "
        f"{json_path}"
    )

    return coco


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n")
    print("=" * 70)
    print("BALL DATASET -> UPDATED LABELS -> COCO")
    print("=" * 70)

    print(
        f"\nSource:\n{SOURCE_ROOT}"
    )

    print(
        f"\nOutput:\n{OUTPUT_ROOT}"
    )

    # ========================================================
    # STEP 1
    # Normalize labels
    # ========================================================

    # Split cases
    split_pairs = create_updated_labels(
        SPLIT_CASES
    )

    # Evaluation case separately
    eval_pairs = create_updated_labels(
        [EVAL_CASE]
    )

    # ========================================================
    # STEP 2
    # Split only the four training cases
    # ========================================================

    print("\n")
    print("=" * 70)
    print("STEP 2: TRAIN / VAL / TEST SPLIT")
    print("=" * 70)

    random.seed(
        RANDOM_SEED
    )

    random.shuffle(
        split_pairs
    )

    total = len(split_pairs)

    train_count = int(
        total * TRAIN_RATIO
    )

    val_count = int(
        total * VAL_RATIO
    )

    test_count = (
        total -
        train_count -
        val_count
    )

    train_pairs = split_pairs[
        :train_count
    ]

    val_pairs = split_pairs[
        train_count:
        train_count + val_count
    ]

    test_pairs = split_pairs[
        train_count + val_count:
    ]

    print(
        f"\nTotal: {total}"
    )

    print(
        f"Train: {len(train_pairs)} "
        f"({len(train_pairs) / total * 100:.2f}%)"
    )

    print(
        f"Val  : {len(val_pairs)} "
        f"({len(val_pairs) / total * 100:.2f}%)"
    )

    print(
        f"Test : {len(test_pairs)} "
        f"({len(test_pairs) / total * 100:.2f}%)"
    )

    # ========================================================
    # STEP 3
    # COCO conversion for train/val/test
    # ========================================================

    coco_split_root = (
        OUTPUT_ROOT /
        "coco_dataset"
    )

    create_coco(
        train_pairs,
        "train",
        coco_split_root
    )

    create_coco(
        val_pairs,
        "val",
        coco_split_root
    )

    create_coco(
        test_pairs,
        "test",
        coco_split_root
    )

    # ========================================================
    # STEP 4
    # COCO conversion for evaluation
    #
    # IMPORTANT:
    # eval is NOT included in the split.
    # ========================================================

    coco_eval_root = (
        OUTPUT_ROOT /
        "coco_eval"
    )

    create_coco(
        eval_pairs,
        "eval",
        coco_eval_root
    )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print("\n")
    print("=" * 70)
    print("FINISHED")
    print("=" * 70)

    print(
        f"""
{OUTPUT_ROOT}/
│
├── updated_labels/
│   ├── AI_train_caseB/
│   ├── basic_train_caseA/
│   ├── cc0_train_caseB/
│   ├── match_train_caseA/
│   └── basic_eval_caseB/
│
├── coco_dataset/
│   ├── images/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   │
│   └── annotations/
│       ├── instances_train.json
│       ├── instances_val.json
│       └── instances_test.json
│
└── coco_eval/
    ├── images/
    │   └── eval/
    │
    └── annotations/
        └── instances_eval.json
"""
    )

    print(
        "\nIMPORTANT:"
    )

    print(
        "basic_eval_caseB was NOT used "
        "for train/val/test."
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()
