import argparse
import json
import shutil
from pathlib import Path

from PIL import Image


# ============================================================
# GENERIC COCO CONVERTER
# ============================================================
#
# INPUT:
#
# images/
#     image1.jpg
#     image2.jpg
#     image3.png
#
# annotations/
#     image1.json
#     image2.json
#     image3.json
#
#
# OUTPUT:
#
# output/
# ├── images/
# │   ├── image1.jpg
# │   ├── image2.jpg
# │   └── image3.png
# │
# └── annotations/
#     └── instances.json
#
#
# USAGE:
#
# python coco_converter.py \
#     --image-dir /path/to/images \
#     --annotation-dir /path/to/annotations \
#     --output-dir /path/to/output
#
#
# If input bbox is XYXY:
#
# [x1, y1, x2, y2]
#
# use:
#
# --bbox-format xyxy
#
#
# If input bbox is already XYWH:
#
# [x, y, width, height]
#
# use:
#
# --bbox-format xywh
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

CATEGORY_ID = 1

CATEGORY_NAME = "ball"

CATEGORY_SUPERCLASS = "object"


# ============================================================
# COMMAND LINE ARGUMENTS
# ============================================================

def parse_arguments():

    parser = argparse.ArgumentParser(
        description=(
            "Convert image and JSON annotations "
            "into a single COCO dataset."
        )
    )

    parser.add_argument(
        "--image-dir",
        required=True,
        type=Path,
        help="Folder containing images."
    )

    parser.add_argument(
        "--annotation-dir",
        required=True,
        type=Path,
        help="Folder containing JSON annotations."
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Folder where COCO output will be created."
    )

    parser.add_argument(
        "--bbox-format",
        choices=[
            "xyxy",
            "xywh",
        ],
        default="xyxy",
        help=(
            "Input bounding box format. "
            "xyxy = [x1,y1,x2,y2], "
            "xywh = [x,y,width,height]. "
            "Default: xyxy"
        )
    )

    parser.add_argument(
        "--category-name",
        default="ball",
        help=(
            "COCO category name. "
            "Default: ball"
        )
    )

    return parser.parse_args()


# ============================================================
# VALIDATE DIRECTORIES
# ============================================================

def validate_directories(args):

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
# FIND CORRESPONDING JSON
# ============================================================

def find_annotation(
    image_path,
    image_dir,
    annotation_dir,
):

    relative_path = image_path.relative_to(
        image_dir
    )

    # --------------------------------------------------------
    # First try matching directory structure.
    #
    # Example:
    #
    # images/
    #   folder1/
    #       image1.jpg
    #
    # annotations/
    #   folder1/
    #       image1.json
    # --------------------------------------------------------

    expected_annotation = (
        annotation_dir /
        relative_path.with_suffix(".json")
    )

    if expected_annotation.exists():

        return expected_annotation

    # --------------------------------------------------------
    # If there is no matching nested structure,
    # search recursively by filename.
    # --------------------------------------------------------

    matches = list(
        annotation_dir.rglob(
            f"{image_path.stem}.json"
        )
    )

    if len(matches) == 1:

        return matches[0]

    if len(matches) > 1:

        print()
        print(
            "[WARNING] Multiple annotations found "
            f"for: {image_path.name}"
        )

        for match in matches:

            print(
                f"          {match}"
            )

        print(
            f"          Using: {matches[0]}"
        )

        return matches[0]

    return None


# ============================================================
# CONVERT XYXY -> XYWH
# ============================================================

def convert_xyxy_to_xywh(rect):

    if not isinstance(rect, list):

        raise ValueError(
            f"Bounding box must be a list: {rect}"
        )

    if len(rect) != 4:

        raise ValueError(
            f"Bounding box must contain "
            f"4 values: {rect}"
        )

    x1, y1, x2, y2 = map(
        float,
        rect
    )

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

def extract_bboxes(
    annotation,
    bbox_format,
):

    data = annotation.get(
        "data",
        {}
    )

    balls = data.get(
        "ball",
        []
    )

    bboxes = []

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
                f"  [WARNING] Invalid bbox skipped: "
                f"{rect}"
            )

            continue

        try:

            if bbox_format == "xyxy":

                bbox = convert_xyxy_to_xywh(
                    rect
                )

            else:

                bbox = list(
                    map(
                        float,
                        rect
                    )
                )

                if (
                    bbox[2] <= 0
                    or bbox[3] <= 0
                ):

                    raise ValueError(
                        f"Invalid XYWH bounding box: "
                        f"{rect}"
                    )

            bboxes.append(
                bbox
            )

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

    dimensions = annotation.get(
        "dimensions"
    )

    # --------------------------------------------------------
    # Use dimensions from annotation if available.
    #
    # Expected:
    #
    # "dimensions": [
    #     height,
    #     width
    # ]
    # --------------------------------------------------------

    if (
        isinstance(
            dimensions,
            list
        )
        and len(dimensions) == 2
    ):

        height = int(
            dimensions[0]
        )

        width = int(
            dimensions[1]
        )

        return width, height

    # --------------------------------------------------------
    # Otherwise read actual image.
    # --------------------------------------------------------

    with Image.open(
        image_path
    ) as image:

        width, height = image.size

    return width, height


# ============================================================
# GENERATE OUTPUT IMAGE NAME
# ============================================================

def get_output_filename(
    image_path,
    image_dir,
):

    relative_path = (
        image_path.relative_to(
            image_dir
        )
    )

    # --------------------------------------------------------
    # Image directly inside input image directory.
    #
    # Example:
    #
    # images/image1.jpg
    #
    # becomes:
    #
    # image1.jpg
    # --------------------------------------------------------

    if len(relative_path.parts) == 1:

        return image_path.name

    # --------------------------------------------------------
    # If nested folders exist, flatten them.
    #
    # Example:
    #
    # images/
    #   folder1/
    #       image1.jpg
    #
    # becomes:
    #
    # folder1_image1.jpg
    # --------------------------------------------------------

    folder_parts = relative_path.parts[:-1]

    filename = relative_path.name

    prefix = "_".join(
        folder_parts
    )

    return (
        f"{prefix}_{filename}"
    )


# ============================================================
# MAIN CONVERSION
# ============================================================

def convert_to_coco(args):

    print()
    print("=" * 75)
    print("GENERIC COCO CONVERSION")
    print("=" * 75)

    print()
    print(
        f"Images      : {args.image_dir}"
    )

    print(
        f"Annotations : {args.annotation_dir}"
    )

    print(
        f"Output      : {args.output_dir}"
    )

    print(
        f"BBox format : {args.bbox_format.upper()}"
    )

    print(
        f"Category    : {args.category_name}"
    )

    # ========================================================
    # CREATE OUTPUT DIRECTORIES
    # ========================================================

    output_images = (
        args.output_dir /
        "images"
    )

    output_annotations = (
        args.output_dir /
        "annotations"
    )

    output_images.mkdir(
        parents=True,
        exist_ok=True
    )

    output_annotations.mkdir(
        parents=True,
        exist_ok=True
    )

    # ========================================================
    # FIND IMAGES
    # ========================================================

    images = find_images(
        args.image_dir
    )

    if not images:

        raise RuntimeError(
            f"No images found in:\n"
            f"{args.image_dir}"
        )

    print()
    print(
        f"Images found: {len(images)}"
    )

    # ========================================================
    # CREATE COCO STRUCTURE
    # ========================================================

    coco = {

        "info": {
            "description":
                "COCO Object Detection Dataset",
            "version":
                "1.0",
        },

        "licenses": [],

        "images": [],

        "annotations": [],

        "categories": [
            {
                "id":
                    CATEGORY_ID,

                "name":
                    args.category_name,

                "supercategory":
                    CATEGORY_SUPERCLASS,
            }
        ],
    }

    # ========================================================
    # IDS
    # ========================================================

    image_id = 1

    annotation_id = 1

    # Used to prevent filename collisions.
    used_filenames = set()

    successful_images = 0

    skipped_images = 0

    missing_annotations = 0

    # ========================================================
    # PROCESS IMAGES
    # ========================================================

    for image_path in images:

        print()
        print(
            f"[{image_id}/{len(images)}] "
            f"{image_path.name}"
        )

        # ----------------------------------------------------
        # Find annotation
        # ----------------------------------------------------

        annotation_path = find_annotation(
            image_path,
            args.image_dir,
            args.annotation_dir,
        )

        if annotation_path is None:

            print(
                "  [MISSING ANNOTATION]"
            )

            missing_annotations += 1

            continue

        # ----------------------------------------------------
        # Read annotation
        # ----------------------------------------------------

        try:

            with open(
                annotation_path,
                "r",
                encoding="utf-8"
            ) as file:

                annotation = json.load(
                    file
                )

        except Exception as e:

            print(
                "  [ERROR] Could not read JSON:"
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
        # Get dimensions
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
                "  [ERROR] Could not get "
                "image dimensions:"
            )

            print(
                f"          {e}"
            )

            skipped_images += 1

            continue

        # ----------------------------------------------------
        # Generate output filename
        # ----------------------------------------------------

        output_filename = (
            get_output_filename(
                image_path,
                args.image_dir,
            )
        )

        # ----------------------------------------------------
        # Prevent filename collisions
        # ----------------------------------------------------

        if output_filename in used_filenames:

            stem = Path(
                output_filename
            ).stem

            suffix = Path(
                output_filename
            ).suffix

            counter = 2

            new_filename = (
                f"{stem}_{counter}"
                f"{suffix}"
            )

            while (
                new_filename
                in used_filenames
            ):

                counter += 1

                new_filename = (
                    f"{stem}_{counter}"
                    f"{suffix}"
                )

            output_filename = (
                new_filename
            )

        used_filenames.add(
            output_filename
        )

        # ----------------------------------------------------
        # Copy image
        # ----------------------------------------------------

        destination = (
            output_images /
            output_filename
        )

        try:

            shutil.copy2(
                image_path,
                destination
            )

        except Exception as e:

            print(
                "  [ERROR] Could not copy image:"
            )

            print(
                f"          {e}"
            )

            skipped_images += 1

            continue

        # ====================================================
        # ADD IMAGE TO COCO
        # ====================================================

        coco["images"].append({

            "id":
                image_id,

            "file_name":
                output_filename,

            "width":
                width,

            "height":
                height,
        })

        # ====================================================
        # GET BOUNDING BOXES
        # ====================================================

        bboxes = extract_bboxes(
            annotation,
            args.bbox_format,
        )

        print(
            f"  Bounding boxes: "
            f"{len(bboxes)}"
        )

        # ====================================================
        # ADD ANNOTATIONS
        # ====================================================

        for bbox in bboxes:

            x = bbox[0]

            y = bbox[1]

            bbox_width = bbox[2]

            bbox_height = bbox[3]

            # ------------------------------------------------
            # Keep bbox inside image boundaries.
            # ------------------------------------------------

            x = max(
                0,
                min(
                    x,
                    width
                )
            )

            y = max(
                0,
                min(
                    y,
                    height
                )
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
                    "  [WARNING] Invalid bbox "
                    "after clipping. Skipped."
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

        # ----------------------------------------------------
        # Next image
        # ----------------------------------------------------

        image_id += 1

        successful_images += 1

    # ========================================================
    # SAVE COCO JSON
    # ========================================================

    json_path = (
        output_annotations /
        "instances.json"
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

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print()
    print("=" * 75)
    print("CONVERSION COMPLETE")
    print("=" * 75)

    print()

    print(
        f"Images found          : "
        f"{len(images)}"
    )

    print(
        f"Images converted      : "
        f"{successful_images}"
    )

    print(
        f"Missing annotations   : "
        f"{missing_annotations}"
    )

    print(
        f"Images skipped        : "
        f"{skipped_images}"
    )

    print(
        f"COCO images           : "
        f"{len(coco['images'])}"
    )

    print(
        f"COCO annotations      : "
        f"{len(coco['annotations'])}"
    )

    print()

    print(
        f"Images output:"
    )

    print(
        f"  {output_images}"
    )

    print()

    print(
        f"COCO JSON:"
    )

    print(
        f"  {json_path}"
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
│   ├── image1.jpg
│   ├── image2.jpg
│   ├── image3.jpg
│   └── ...
│
└── annotations/
    └── instances.json
"""
    )

    print(
        "Original images and annotations "
        "were not modified."
    )


# ============================================================
# RUN
# ============================================================

def main():

    args = parse_arguments()

    validate_directories(
        args
    )

    convert_to_coco(
        args
    )


if __name__ == "__main__":
    main()
