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

# This case is ONLY converted to YOLO.
# It is NOT used for train/val/test.
EVAL_CASE = "basic_eval_caseB"


# Cases whose original annotations are XYXY
XYXY_CASES = {
    "AI_train_caseB",
    "basic_train_caseA",
    "basic_eval_caseB",
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
# YOLO CLASSES
# ============================================================

# YOLO class ids are 0-indexed, in list order.
CLASS_NAMES = [
    "ball",
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
    Convert an annotation to the common XYWH format
    (still in pixel units, top-left x/y + width/height).

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
# CREATE YOLO DATASET
# ============================================================

def create_yolo(
    pairs,
    split_name,
    output_root,
):
    """
    Convert normalized XYWH (pixel) annotations into YOLO format.

    Output layout:

        output_root/
            images/<split_name>/*.jpg
            labels/<split_name>/*.txt

    Each label line:

        class_id x_center y_center width height

    All values normalized to [0, 1] relative to image width/height.
    """

    print("\n")
    print("=" * 70)
    print(f"YOLO CONVERSION: {split_name.upper()}")
    print("=" * 70)

    images_output_dir = (
        output_root /
        "images" /
        split_name
    )

    labels_output_dir = (
        output_root /
        "labels" /
        split_name
    )

    images_output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    labels_output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    num_images = 0
    num_annotations = 0

    # Class name -> class id (0-indexed, as YOLO expects)
    class_to_id = {
        name: idx
        for idx, name in enumerate(CLASS_NAMES)
    }

    for item in pairs:

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

        if width <= 0 or height <= 0:

            print(
                f"[WARNING] Bad image dimensions "
                f"for {image_path}: {width}x{height}"
            )

            continue

        # ----------------------------------------------------
        # Unique image/label filename
        # ----------------------------------------------------

        unique_name = get_unique_image_name(
            case_name,
            imgs_dir,
            image_path.name
        )

        unique_stem = Path(unique_name).stem

        image_destination = (
            images_output_dir /
            unique_name
        )

        label_destination = (
            labels_output_dir /
            f"{unique_stem}.txt"
        )

        # ----------------------------------------------------
        # Ball annotations -> YOLO lines
        # ----------------------------------------------------

        data = annotation_data.get(
            "data",
            {}
        )

        balls = data.get(
            "ball",
            []
        )

        yolo_lines = []

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
            # At this point rect is already XYWH (pixels).

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
            # Convert to normalized YOLO format
            # (class_id, x_center, y_center, width, height)
            # ------------------------------------------------

            x_center = (x + bbox_width / 2.0) / width
            y_center = (y + bbox_height / 2.0) / height
            norm_width = bbox_width / width
            norm_height = bbox_height / height

            # Clip to [0, 1] in case of rounding/edge boxes
            x_center = min(max(x_center, 0.0), 1.0)
            y_center = min(max(y_center, 0.0), 1.0)
            norm_width = min(max(norm_width, 0.0), 1.0)
            norm_height = min(max(norm_height, 0.0), 1.0)

            class_id = class_to_id["ball"]

            yolo_lines.append(
                f"{class_id} "
                f"{x_center:.6f} "
                f"{y_center:.6f} "
                f"{norm_width:.6f} "
                f"{norm_height:.6f}"
            )

            num_annotations += 1

        # ----------------------------------------------------
        # Copy image + write label file
        # (write label even if empty, so YOLO treats it as
        #  a negative/background image rather than "missing")
        # ----------------------------------------------------

        shutil.copy2(
            image_path,
            image_destination
        )

        with open(
            label_destination,
            "w",
            encoding="utf-8"
        ) as f:

            f.write(
                "\n".join(yolo_lines)
            )

            if yolo_lines:
                f.write("\n")

        num_images += 1

    print(
        f"\nImages      : "
        f"{num_images}"
    )

    print(
        f"Annotations : "
        f"{num_annotations}"
    )

    print(
        f"Images saved to : "
        f"{images_output_dir}"
    )

    print(
        f"Labels saved to : "
        f"{labels_output_dir}"
    )

    return {
        "images": num_images,
        "annotations": num_annotations,
    }


def write_data_yaml(output_root, splits):
    """
    Write a YOLO-style data.yaml describing this dataset.

    `splits` is a dict like {"train": True, "val": True, "test": True}
    indicating which split subfolders exist under output_root.
    """

    yaml_path = output_root / "data.yaml"

    lines = [
        f"path: {output_root}",
    ]

    for split_name in ("train", "val", "test"):

        if splits.get(split_name):

            lines.append(
                f"{split_name}: images/{split_name}"
            )

    lines.append("")
    lines.append(f"nc: {len(CLASS_NAMES)}")
    lines.append(
        "names: ["
        + ", ".join(f"'{name}'" for name in CLASS_NAMES)
        + "]"
    )

    with open(
        yaml_path,
        "w",
        encoding="utf-8"
    ) as f:

        f.write("\n".join(lines) + "\n")

    print(
        f"\ndata.yaml saved to: {yaml_path}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n")
    print("=" * 70)
    print("BALL DATASET -> UPDATED LABELS -> YOLO")
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
    # YOLO conversion for train/val/test
    # ========================================================

    yolo_split_root = (
        OUTPUT_ROOT /
        "yolo_dataset"
    )

    create_yolo(
        train_pairs,
        "train",
        yolo_split_root
    )

    create_yolo(
        val_pairs,
        "val",
        yolo_split_root
    )

    create_yolo(
        test_pairs,
        "test",
        yolo_split_root
    )

    write_data_yaml(
        yolo_split_root,
        {"train": True, "val": True, "test": True}
    )

    # ========================================================
    # STEP 4
    # YOLO conversion for evaluation
    #
    # IMPORTANT:
    # eval is NOT included in the split.
    # ========================================================

    yolo_eval_root = (
        OUTPUT_ROOT /
        "yolo_eval"
    )

    create_yolo(
        eval_pairs,
        "eval",
        yolo_eval_root
    )

    # eval isn't train/val/test, so write a minimal yaml
    # pointing at it separately (useful for standalone
    # YOLO val runs against this set).
    eval_yaml_path = yolo_eval_root / "data.yaml"

    with open(
        eval_yaml_path,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            f"path: {yolo_eval_root}\n"
            f"val: images/eval\n"
            f"\n"
            f"nc: {len(CLASS_NAMES)}\n"
            f"names: ["
            + ", ".join(f"'{name}'" for name in CLASS_NAMES)
            + "]\n"
        )

    print(
        f"\ndata.yaml saved to: {eval_yaml_path}"
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
├── yolo_dataset/
│   ├── images/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   │
│   ├── labels/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   │
│   └── data.yaml
│
└── yolo_eval/
    ├── images/
    │   └── eval/
    │
    ├── labels/
    │   └── eval/
    │
    └── data.yaml
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
