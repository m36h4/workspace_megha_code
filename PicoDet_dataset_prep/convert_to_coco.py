import json
import random
import shutil
from pathlib import Path
from PIL import Image


# ============================================================
# CONFIGURATION
# ============================================================

# ------------------------------------------------------------
# CHANGE THIS TO YOUR DATASET LOCATION
# ------------------------------------------------------------

SOURCE_ROOT = Path("/home/eng_megha/balldataset")


# ------------------------------------------------------------
# ALL GENERATED DATA WILL GO HERE
# ------------------------------------------------------------

OUTPUT_ROOT = Path("/home/eng_megha/balldataset_coco")


# ============================================================
# DATASET CASES
# ============================================================

# These four cases are used ONLY for the 70/20/10 split.
SPLIT_CASES = [
    "AI_train_caseB",
    "basic_train_caseA",
    "cc0_train_caseB",
    "match_train_caseA",
]


# ------------------------------------------------------------
# EVALUATION CASE
# ------------------------------------------------------------

# This case is NOT used in the 70/20/10 split.
#
# It is:
#
# 1. Converted separately into coco_eval
# 2. Added to merge_val together with the 20% validation set
#
EVAL_CASE = "basic_eval_caseA"


# ============================================================
# ANNOTATION FORMATS
# ============================================================

# These cases contain:
#
# [x1, y1, x2, y2]
#
# and need conversion to:
#
# [x, y, width, height]

XYXY_CASES = {
    "AI_train_caseB",
    "basic_train_caseA",
    "basic_eval_caseA",
}


# ------------------------------------------------------------
# These cases already contain:
#
# [x, y, width, height]
#
# so the bbox is copied without changing it.
# ------------------------------------------------------------

XYWH_CASES = {
    "cc0_train_caseB",
    "match_train_caseA",
}


# ============================================================
# FOLDERS TO EXCLUDE
# ============================================================

# These folders will NEVER be processed.

EXCLUDED_FOLDERS = {
    "aug",
    "other",
    "volleyball",
}


# ============================================================
# SPLIT RATIOS
# ============================================================

TRAIN_RATIO = 0.70
VAL_RATIO = 0.20
TEST_RATIO = 0.10


# Fixed seed means you get the same split every time.

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
# VALIDATE SPLIT RATIOS
# ============================================================

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
# HELPER FUNCTIONS
# ============================================================

def is_excluded(path):
    """
    Check whether any directory in the path is one of
    the excluded folders.

    Example:

        basic_train_caseA/volleyball/imgs

    will be excluded.
    """

    return any(
        part.lower() in EXCLUDED_FOLDERS
        for part in path.parts
    )


# ------------------------------------------------------------

def get_annotation_format(case_name):
    """
    Return the annotation format for a particular case.
    """

    if case_name in XYXY_CASES:
        return "xyxy"

    if case_name in XYWH_CASES:
        return "xywh"

    raise ValueError(
        f"Annotation format is not defined for "
        f"case: {case_name}"
    )


# ------------------------------------------------------------

def find_imgs_directories(case_root):
    """
    Find every 'imgs' directory recursively.

    Example:

        basic_train_caseA/
            basketball/
                imgs/
            rugby/
                imgs/
            soccer/
                imgs/

    All three imgs directories will be found.

    Excluded directories are ignored.
    """

    imgs_dirs = []

    for path in case_root.rglob("imgs"):

        if not path.is_dir():
            continue

        if is_excluded(path):
            continue

        imgs_dirs.append(path)

    return sorted(imgs_dirs)


# ------------------------------------------------------------

def convert_xyxy_to_xywh(rect):
    """
    Convert:

        [x1, y1, x2, y2]

    to:

        [x, y, width, height]
    """

    if not isinstance(rect, list):
        raise ValueError(
            f"Bounding box must be a list: {rect}"
        )

    if len(rect) != 4:
        raise ValueError(
            f"Bounding box must have 4 values: {rect}"
        )

    x1, y1, x2, y2 = rect

    width = x2 - x1
    height = y2 - y1

    if width < 0 or height < 0:
        raise ValueError(
            f"Invalid XYXY bounding box: {rect}"
        )

    return [
        x1,
        y1,
        width,
        height,
    ]


# ------------------------------------------------------------

def normalize_annotation(
    annotation,
    annotation_format,
):
    """
    Convert an annotation into the common XYWH format.

    XYXY:
        [x1, y1, x2, y2]
        ->
        [x, y, width, height]

    XYWH:
        [x, y, width, height]
        ->
        unchanged

    The original JSON object is not modified.
    """

    # Deep copy
    annotation = json.loads(
        json.dumps(annotation)
    )

    data = annotation.get(
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

        if annotation_format == "xyxy":

            entire["rect"] = (
                convert_xyxy_to_xywh(rect)
            )

        elif annotation_format == "xywh":

            # Already in desired format.
            entire["rect"] = rect

        else:

            raise ValueError(
                f"Unknown annotation format: "
                f"{annotation_format}"
            )

    return annotation


# ------------------------------------------------------------

def get_unique_image_name(
    case_name,
    imgs_dir,
    image_name,
):
    """
    Generate a unique image filename.

    This prevents collisions when different folders contain
    images with the same filename.

    Example:

        AI_train_caseB_basketball_image1.jpg

        basic_eval_caseA_basketball_image1.jpg
    """

    case_root = SOURCE_ROOT / case_name

    try:

        relative_path = (
            imgs_dir.relative_to(case_root)
        )

        parts = relative_path.parts

        # Remove imgs from the path
        parts = [
            p
            for p in parts
            if p.lower() != "imgs"
        ]

        prefix = "_".join(parts)

    except ValueError:

        prefix = imgs_dir.parent.name

    prefix = prefix.replace(
        "/",
        "_"
    )

    prefix = prefix.replace(
        "\\",
        "_"
    )

    if prefix:

        return (
            f"{case_name}_"
            f"{prefix}_"
            f"{image_name}"
        )

    return (
        f"{case_name}_"
        f"{image_name}"
    )


# ============================================================
# STEP 1
# CREATE UPDATED LABELS
# ============================================================

def create_updated_labels(case_names):
    """
    Create normalized labels.

    XYXY -> XYWH

    XYWH -> copied unchanged

    Original labels are NEVER modified.

    Returns a list of image/label pairs.
    """

    print()
    print("=" * 75)
    print("STEP 1: CREATING UPDATED LABELS")
    print("=" * 75)

    updated_root = (
        OUTPUT_ROOT /
        "updated_labels"
    )

    updated_root.mkdir(
        parents=True,
        exist_ok=True
    )

    all_pairs = []

    for case_name in case_names:

        case_root = (
            SOURCE_ROOT /
            case_name
        )

        # ----------------------------------------------------
        # Check case exists
        # ----------------------------------------------------

        if not case_root.exists():

            print(
                "\n[WARNING] Case not found:"
            )

            print(
                f"          {case_root}"
            )

            continue

        annotation_format = (
            get_annotation_format(
                case_name
            )
        )

        print()
        print(
            f"Processing: {case_name}"
        )

        print(
            f"Format: {annotation_format.upper()}"
        )

        # ----------------------------------------------------
        # Find imgs directories
        # ----------------------------------------------------

        imgs_dirs = (
            find_imgs_directories(
                case_root
            )
        )

        if not imgs_dirs:

            print(
                "  [WARNING] No imgs directories found."
            )

            continue

        # ----------------------------------------------------
        # Process every imgs directory
        # ----------------------------------------------------

        for imgs_dir in imgs_dirs:

            labels_dir = (
                imgs_dir.parent /
                "labels"
            )

            if not labels_dir.exists():

                print(
                    "\n  [WARNING] Labels directory "
                    "not found:"
                )

                print(
                    f"             {labels_dir}"
                )

                continue

            print()
            print(
                f"  Images : {imgs_dir}"
            )

            print(
                f"  Labels : {labels_dir}"
            )

            # ------------------------------------------------
            # Preserve original directory structure
            # ------------------------------------------------

            relative_parent = (
                imgs_dir.parent.relative_to(
                    case_root
                )
            )

            output_labels_dir = (
                updated_root /
                case_name /
                relative_parent /
                "labels"
            )

            output_labels_dir.mkdir(
                parents=True,
                exist_ok=True
            )

            # ------------------------------------------------
            # Process images
            # ------------------------------------------------

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

                # ------------------------------------------------
                # Find corresponding JSON
                # ------------------------------------------------

                label_path = (
                    labels_dir /
                    f"{image_path.stem}.json"
                )

                if not label_path.exists():

                    print(
                        f"    [MISSING LABEL] "
                        f"{image_path.name}"
                    )

                    continue

                # ------------------------------------------------
                # Read annotation
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
                        "    [ERROR] Could not read:"
                    )

                    print(
                        f"            {label_path}"
                    )

                    print(
                        f"            {e}"
                    )

                    continue

                # ------------------------------------------------
                # Normalize annotation
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
                        "    [ERROR] Could not "
                        "normalize:"
                    )

                    print(
                        f"            {label_path}"
                    )

                    print(
                        f"            {e}"
                    )

                    continue

                # ------------------------------------------------
                # Save updated annotation
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

                # ------------------------------------------------
                # Store pair information
                # ------------------------------------------------

                all_pairs.append({
                    "case": case_name,
                    "image": image_path,
                    "updated_label": (
                        output_label_path
                    ),
                    "imgs_dir": imgs_dir,
                })

    print()
    print(
        "-" * 75
    )

    print(
        f"Total image-label pairs: "
        f"{len(all_pairs)}"
    )

    print(
        "Updated labels saved at:"
    )

    print(
        f"  {updated_root}"
    )

    return all_pairs


# ============================================================
# STEP 2
# CREATE COCO DATASET
# ============================================================

def create_coco(
    pairs,
    dataset_name,
    split_name,
    output_root,
):
    """
    Convert normalized XYWH annotations into COCO format.

    Parameters:

        pairs
            image/annotation pairs

        dataset_name
            informational name

        split_name
            train / val / test / eval / merge_val

        output_root
            destination root
    """

    print()
    print("=" * 75)
    print(
        f"COCO CONVERSION: "
        f"{dataset_name} / {split_name}"
    )
    print("=" * 75)

    # --------------------------------------------------------
    # Output directories
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # COCO structure
    # --------------------------------------------------------

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

    annotation_id = 1

    successful_images = 0
    skipped_images = 0

    # --------------------------------------------------------
    # Process each image
    # --------------------------------------------------------

    for image_id, item in enumerate(
        pairs,
        start=1
    ):

        image_path = item["image"]

        label_path = item[
            "updated_label"
        ]

        case_name = item[
            "case"
        ]

        imgs_dir = item[
            "imgs_dir"
        ]

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
                "[ERROR] Reading:"
            )

            print(
                f"        {label_path}"
            )

            print(
                f"        {e}"
            )

            skipped_images += 1

            continue

        # ----------------------------------------------------
        # Get image dimensions
        # ----------------------------------------------------

        dimensions = (
            annotation_data.get(
                "dimensions"
            )
        )

        if (
            dimensions is not None
            and len(dimensions) == 2
        ):

            # Your annotation structure is:
            #
            # "dimensions": [
            #     848,
            #     1264
            # ]
            #
            # Interpreted as:
            #
            # height = 848
            # width  = 1264

            height = int(
                dimensions[0]
            )

            width = int(
                dimensions[1]
            )

        else:

            # ------------------------------------------------
            # Fallback to actual image dimensions
            # ------------------------------------------------

            try:

                with Image.open(
                    image_path
                ) as im:

                    width, height = (
                        im.size
                    )

            except Exception as e:

                print(
                    "[ERROR] Cannot determine "
                    "image size:"
                )

                print(
                    f"        {image_path}"
                )

                print(
                    f"        {e}"
                )

                skipped_images += 1

                continue

        # ----------------------------------------------------
        # Generate unique filename
        # ----------------------------------------------------

        unique_name = (
            get_unique_image_name(
                case_name,
                imgs_dir,
                image_path.name
            )
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
                "[ERROR] Copying:"
            )

            print(
                f"        {image_path}"
            )

            print(
                f"        {e}"
            )

            skipped_images += 1

            continue

        # ----------------------------------------------------
        # Add COCO image
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
        # Get ball annotations
        # ----------------------------------------------------

        data = annotation_data.get(
            "data",
            {}
        )

        balls = data.get(
            "ball",
            []
        )

        # ----------------------------------------------------
        # Add every ball
        # ----------------------------------------------------

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

            if (
                not isinstance(
                    rect,
                    list
                )
                or len(rect) != 4
            ):

                print(
                    "[WARNING] Invalid bbox:"
                )

                print(
                    f"            {rect}"
                )

                print(
                    f"            {label_path}"
                )

                continue

            # ------------------------------------------------
            # IMPORTANT:
            #
            # At this point ALL labels have already been
            # normalized to XYWH.
            #
            # Therefore:
            #
            # rect = [x, y, width, height]
            # ------------------------------------------------

            x = float(
                rect[0]
            )

            y = float(
                rect[1]
            )

            bbox_width = float(
                rect[2]
            )

            bbox_height = float(
                rect[3]
            )

            # ------------------------------------------------
            # Validate bbox
            # ------------------------------------------------

            if (
                bbox_width <= 0
                or bbox_height <= 0
            ):

                print(
                    "[WARNING] Invalid "
                    "bbox dimensions:"
                )

                print(
                    f"            {rect}"
                )

                print(
                    f"            {label_path}"
                )

                continue

            # ------------------------------------------------
            # COCO annotation
            # ------------------------------------------------

            coco["annotations"].append({

                "id":
                    annotation_id,

                "image_id":
                    image_id,

                "category_id":
                    1,

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

        successful_images += 1

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

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print()
    print(
        f"Images copied      : "
        f"{successful_images}"
    )

    print(
        f"Images skipped     : "
        f"{skipped_images}"
    )

    print(
        f"COCO annotations   : "
        f"{len(coco['annotations'])}"
    )

    print(
        f"JSON saved to      : "
        f"{json_path}"
    )

    return coco


# ============================================================
# STEP 3
# CREATE MERGED VALIDATION SET
# ============================================================

def create_merge_val(
    val_pairs,
    eval_pairs,
    output_root,
):
    """
    Create:

        merge_val =
            normal 20% validation
            +
            basic_eval_caseA

    IMPORTANT:

        The normal val set is NOT changed.

        basic_eval_caseA is added only to merge_val.
    """

    print()
    print("=" * 75)
    print("CREATING MERGE_VAL")
    print("=" * 75)

    print()
    print(
        f"20% split validation : "
        f"{len(val_pairs)} images"
    )

    print(
        f"basic_eval_caseA    : "
        f"{len(eval_pairs)} images"
    )

    # --------------------------------------------------------
    # Combine
    # --------------------------------------------------------

    merge_pairs = (
        val_pairs +
        eval_pairs
    )

    print(
        f"merge_val total      : "
        f"{len(merge_pairs)} images"
    )

    # --------------------------------------------------------
    # Print distribution
    # --------------------------------------------------------

    print_case_distribution(
        merge_pairs,
        "MERGE_VAL"
    )

    # --------------------------------------------------------
    # Convert to COCO
    # --------------------------------------------------------

    return create_coco(
        merge_pairs,
        "Merged Validation",
        "merge_val",
        output_root,
    )


# ============================================================
# STEP 4
# PRINT CASE DISTRIBUTION
# ============================================================

def print_case_distribution(
    pairs,
    name,
):
    """
    Print how many images from each case are present.
    """

    counts = {}

    for item in pairs:

        case = item["case"]

        counts[case] = (
            counts.get(case, 0) +
            1
        )

    print()
    print(
        f"{name} distribution:"
    )

    for case, count in sorted(
        counts.items()
    ):

        print(
            f"  {case}: {count}"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 75)
    print("BALL DATASET PROCESSING")
    print("=" * 75)

    print()
    print(
        "Source dataset:"
    )

    print(
        f"  {SOURCE_ROOT}"
    )

    print()
    print(
        "Output directory:"
    )

    print(
        f"  {OUTPUT_ROOT}"
    )

    # ========================================================
    # CHECK SOURCE
    # ========================================================

    if not SOURCE_ROOT.exists():

        raise FileNotFoundError(
            f"\nSource dataset does not exist:\n"
            f"{SOURCE_ROOT}\n\n"
            f"Please change SOURCE_ROOT at the "
            f"top of the script."
        )

    # ========================================================
    # PRINT CONFIGURATION
    # ========================================================

    print()
    print(
        "Cases used for 70/20/10 split:"
    )

    for case in SPLIT_CASES:

        fmt = get_annotation_format(
            case
        )

        print(
            f"  + {case} "
            f"[{fmt.upper()}]"
        )

    print()
    print(
        "Separate evaluation case:"
    )

    print(
        f"  + {EVAL_CASE} "
        f"[{get_annotation_format(EVAL_CASE).upper()}]"
    )

    print()
    print(
        "Excluded folders:"
    )

    for folder in sorted(
        EXCLUDED_FOLDERS
    ):

        print(
            f"  - {folder}"
        )

    # ========================================================
    # STEP 1A
    #
    # Normalize the four split cases
    # ========================================================

    split_pairs = create_updated_labels(
        SPLIT_CASES
    )

    # ========================================================
    # STEP 1B
    #
    # Normalize basic_eval_caseA separately
    # ========================================================

    eval_pairs = create_updated_labels(
        [EVAL_CASE]
    )

    # ========================================================
    # CHECK DATA
    # ========================================================

    if len(split_pairs) == 0:

        raise RuntimeError(
            "\nNo training image-label pairs "
            "were found.\n"
            "Check your SOURCE_ROOT and "
            "dataset structure."
        )

    if len(eval_pairs) == 0:

        print()
        print(
            "[WARNING] No basic_eval_caseA "
            "image-label pairs were found."
        )

    # ========================================================
    # STEP 2
    #
    # RANDOM 70 / 20 / 10 SPLIT
    #
    # IMPORTANT:
    #
    # basic_eval_caseA is NOT included here.
    # ========================================================

    print()
    print("=" * 75)
    print("STEP 2: 70 / 20 / 10 SPLIT")
    print("=" * 75)

    random.seed(
        RANDOM_SEED
    )

    random.shuffle(
        split_pairs
    )

    total = len(
        split_pairs
    )

    # --------------------------------------------------------
    # Calculate counts
    # --------------------------------------------------------

    train_count = int(
        total *
        TRAIN_RATIO
    )

    val_count = int(
        total *
        VAL_RATIO
    )

    # Test receives everything remaining.
    test_count = (
        total -
        train_count -
        val_count
    )

    # --------------------------------------------------------
    # Create splits
    # --------------------------------------------------------

    train_pairs = split_pairs[
        :train_count
    ]

    val_pairs = split_pairs[
        train_count:
        train_count +
        val_count
    ]

    test_pairs = split_pairs[
        train_count +
        val_count:
    ]

    # ========================================================
    # PRINT SPLIT SUMMARY
    # ========================================================

    print()

    print(
        f"Total split images : "
        f"{total}"
    )

    print()

    print(
        f"Train              : "
        f"{len(train_pairs)} "
        f"({len(train_pairs) / total * 100:.2f}%)"
    )

    print(
        f"Val                : "
        f"{len(val_pairs)} "
        f"({len(val_pairs) / total * 100:.2f}%)"
    )

    print(
        f"Test               : "
        f"{len(test_pairs)} "
        f"({len(test_pairs) / total * 100:.2f}%)"
    )

    print()

    print(
        "NOTE:"
    )

    print(
        "basic_eval_caseA is NOT included "
        "in these three splits."
    )

    print_case_distribution(
        train_pairs,
        "TRAIN"
    )

    print_case_distribution(
        val_pairs,
        "VAL"
    )

    print_case_distribution(
        test_pairs,
        "TEST"
    )

    # ========================================================
    # STEP 3
    #
    # CREATE NORMAL COCO TRAIN / VAL / TEST
    #
    # val contains ONLY the 20% split.
    # ========================================================

    coco_dataset_root = (
        OUTPUT_ROOT /
        "coco_dataset"
    )

    print()
    print("=" * 75)
    print("STEP 3: COCO TRAIN / VAL / TEST")
    print("=" * 75)

    create_coco(
        train_pairs,
        "Training Dataset",
        "train",
        coco_dataset_root,
    )

    create_coco(
        val_pairs,
        "Validation Dataset",
        "val",
        coco_dataset_root,
    )

    create_coco(
        test_pairs,
        "Test Dataset",
        "test",
        coco_dataset_root,
    )

    # ========================================================
    # STEP 4
    #
    # BASIC EVAL CASE
    #
    # Completely separate COCO dataset.
    #
    # Contains ONLY basic_eval_caseA.
    # ========================================================

    coco_eval_root = (
        OUTPUT_ROOT /
        "coco_eval"
    )

    print()
    print("=" * 75)
    print("STEP 4: BASIC EVAL COCO")
    print("=" * 75)

    if eval_pairs:

        create_coco(
            eval_pairs,
            "Basic Evaluation Dataset",
            "eval",
            coco_eval_root,
        )

    # ========================================================
    # STEP 5
    #
    # MERGED VALIDATION
    #
    # 20% validation + basic_eval_caseA
    #
    # IMPORTANT:
    #
    # This does NOT modify coco_dataset/val.
    # ========================================================

    coco_merge_val_root = (
        OUTPUT_ROOT /
        "coco_merge_val"
    )

    print()
    print("=" * 75)
    print("STEP 5: MERGED VALIDATION")
    print("=" * 75)

    create_merge_val(
        val_pairs,
        eval_pairs,
        coco_merge_val_root,
    )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print()
    print("=" * 75)
    print("PROCESSING COMPLETE")
    print("=" * 75)

    print()
    print(
        "Output structure:"
    )

    print(
        f"""
{OUTPUT_ROOT}/
│
├── updated_labels/
│   ├── AI_train_caseB/
│   ├── basic_train_caseA/
│   ├── cc0_train_caseB/
│   ├── match_train_caseA/
│   └── basic_eval_caseA/
│
├── coco_dataset/
│   │
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
├── coco_eval/
│   │
│   ├── images/
│   │   └── eval/
│   │
│   └── annotations/
│       └── instances_eval.json
│
└── coco_merge_val/
    │
    ├── images/
    │   └── merge_val/
    │
    └── annotations/
        └── instances_merge_val.json
"""
    )

    # ========================================================
    # FINAL DATASET LOGIC
    # ========================================================

    print(
        "Dataset logic:"
    )

    print()

    print(
        f"  Train     = "
        f"{len(train_pairs)} images"
    )

    print(
        f"  Val       = "
        f"{len(val_pairs)} images "
        f"(20% split ONLY)"
    )

    print(
        f"  Test      = "
        f"{len(test_pairs)} images"
    )

    print(
        f"  Eval      = "
        f"{len(eval_pairs)} images "
        f"(basic_eval_caseA ONLY)"
    )

    print(
        f"  Merge Val = "
        f"{len(val_pairs) + len(eval_pairs)} images "
        f"(20% val + basic_eval_caseA)"
    )

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "  basic_eval_caseA was NOT used "
        "in the 70/20/10 split."
    )

    print(
        "  coco_dataset/val contains ONLY "
        "the 20% validation split."
    )

    print(
        "  coco_eval/eval contains ONLY "
        "basic_eval_caseA."
    )

    print(
        "  coco_merge_val/merge_val contains "
        "20% validation + basic_eval_caseA."
    )

    print()
    print(
        "Original dataset was NOT modified."
    )

    print()
    print("=" * 75)
    print("DONE")
    print("=" * 75)


# ============================================================
# RUN SCRIPT
# ============================================================

if __name__ == "__main__":
    main()
