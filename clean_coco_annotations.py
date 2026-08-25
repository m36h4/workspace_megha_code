#!/usr/bin/env python3
"""
COCO Annotation Cleaner

Expected structure:

dataset/
├── train/
│   ├── *.jpg
│   └── _annotations.json
├── test/
│   ├── *.jpg
│   └── _annotations.json
├── val/
│   ├── *.jpg
│   └── _annotations.json
└── val_eval/
    ├── *.jpg
    └── _annotations.json

COCO bbox format:
    [x, y, width, height]

The script:
- Processes train/test/val/val_eval.
- Never changes the original dataset.
- Copies images and cleaned _annotations.json to a new output folder.
- Checks every annotation against the ACTUAL image dimensions.
- Clips negative/out-of-bound boxes where possible.
- Removes boxes that are completely outside the image.
- Removes zero/negative-size boxes.
- Reports images with no remaining annotations separately.
- Updates COCO image width/height to actual image dimensions.
- Updates COCO annotation area after bbox correction.
- Produces CSV audit reports.

Usage:
    python clean_coco_annotations.py --root /path/to/dataset --out /path/to/cleaned_dataset

Recommended first run:
    python clean_coco_annotations.py --root /path/to/dataset --out /path/to/cleaned_dataset --dry-run

Then, after checking the report:
    python clean_coco_annotations.py --root /path/to/dataset --out /path/to/cleaned_dataset
"""

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path
from collections import Counter, defaultdict

try:
    from PIL import Image
except ImportError:
    print("ERROR: Pillow is required.")
    print("Install it with: pip install pillow")
    sys.exit(1)


SPLITS = ["train", "test", "val", "val_eval"]
ANNOTATION_FILE = "_annotations.json"
IMAGE_EXTS = {
    ".jpg", ".jpeg", ".png", ".bmp",
    ".tif", ".tiff", ".webp"
}


def get_image_dimensions(image_path):
    """Read actual image dimensions without loading the full image."""
    try:
        with Image.open(image_path) as img:
            return img.width, img.height, None
    except Exception as e:
        return None, None, str(e)


def find_image(split_dir, file_name):
    """
    Locate an image referenced by COCO file_name.

    Supports:
        image.jpg
        folder/image.jpg
        nested/folder/image.jpg

    Your current dataset normally has images directly inside
    train/test/val/val_eval.
    """
    if not file_name:
        return None

    normalized = str(file_name).replace("\\", "/").lstrip("/")

    # Exact relative path.
    candidate = split_dir / normalized
    if candidate.is_file():
        return candidate

    # Basename directly under split.
    basename = Path(normalized).name
    candidate = split_dir / basename
    if candidate.is_file():
        return candidate

    # Recursive fallback.
    matches = list(split_dir.rglob(basename))

    if len(matches) == 1:
        return matches[0]

    if len(matches) > 1:
        # Deterministic fallback.
        return sorted(matches)[0]

    return None


def normalize_number(value):
    """Make COCO JSON cleaner: 10.0 -> 10."""
    value = float(value)

    if abs(value - round(value)) < 1e-9:
        return int(round(value))

    return round(value, 6)


def clean_bbox(bbox, img_w, img_h):
    """
    Validate and correct one COCO bbox.

    Input:
        [x, y, width, height]

    Returns:
        corrected_bbox
        issues
        action
        original_bbox

    Possible actions:
        unchanged
        clipped
        removed
    """

    original_bbox = bbox

    if not isinstance(bbox, list) or len(bbox) != 4:
        return (
            None,
            ["invalid_bbox_format"],
            "removed",
            original_bbox
        )

    try:
        x, y, width, height = [
            float(v) for v in bbox
        ]
    except (TypeError, ValueError):
        return (
            None,
            ["invalid_bbox_values"],
            "removed",
            original_bbox
        )

    issues = []

    # Invalid original dimensions.
    if width <= 0 or height <= 0:
        issues.append("zero_or_negative_size")

    # Coordinates of bottom-right corner.
    x2 = x + width
    y2 = y + height

    # Negative coordinates.
    if x < 0 or y < 0:
        issues.append("negative_coordinate")

    # Beyond image bounds.
    if x2 > img_w or y2 > img_h:
        issues.append("exceeds_image_bounds")

    # Completely outside image.
    if x2 <= 0 or y2 <= 0 or x >= img_w or y >= img_h:
        issues.append("bbox_outside_image")
        return (
            None,
            sorted(set(issues)),
            "removed",
            original_bbox
        )

    # Don't manufacture a valid bbox from an invalid original size.
    if width <= 0 or height <= 0:
        return (
            None,
            sorted(set(issues)),
            "removed",
            original_bbox
        )

    # Convert to corners, clip, then convert back to COCO xywh.
    new_x1 = max(0.0, x)
    new_y1 = max(0.0, y)

    new_x2 = min(float(img_w), x2)
    new_y2 = min(float(img_h), y2)

    new_width = new_x2 - new_x1
    new_height = new_y2 - new_y1

    # Box became invalid after clipping.
    if new_width <= 0 or new_height <= 0:
        issues.append("zero_or_negative_size")
        return (
            None,
            sorted(set(issues)),
            "removed",
            original_bbox
        )

    corrected = [
        normalize_number(new_x1),
        normalize_number(new_y1),
        normalize_number(new_width),
        normalize_number(new_height)
    ]

    if corrected != original_bbox:
        action = "clipped"
    else:
        action = "unchanged"

    return (
        corrected,
        sorted(set(issues)),
        action,
        original_bbox
    )


def copy_images(src_dir, dst_dir):
    """Copy all images while preserving any relative structure."""
    copied = 0

    for path in src_dir.rglob("*"):
        if not path.is_file():
            continue

        if path.name == ANNOTATION_FILE:
            continue

        if path.suffix.lower() not in IMAGE_EXTS:
            continue

        relative = path.relative_to(src_dir)
        target = dst_dir / relative
        target.parent.mkdir(parents=True, exist_ok=True)

        shutil.copy2(path, target)
        copied += 1

    return copied


def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fields,
            extrasaction="ignore"
        )

        writer.writeheader()
        writer.writerows(rows)


def process_split(
    src_root,
    dst_root,
    split,
    dry_run=False,
    copy_image_files=True
):
    src_dir = src_root / split
    dst_dir = dst_root / split
    annotation_path = src_dir / ANNOTATION_FILE

    result = {
        "split": split,
        "status": "SKIPPED",
        "images": 0,
        "annotations_before": 0,
        "annotations_after": 0,
        "annotations_clipped": 0,
        "annotations_removed": 0,
        "empty_label_images": 0,
        "missing_images": 0,
        "image_read_errors": 0,
        "issues": Counter(),
        "changes": [],
        "empty_images": [],
        "missing_image_rows": [],
        "dimension_rows": []
    }

    if not src_dir.exists():
        result["status"] = "MISSING_SPLIT"
        return result

    if not annotation_path.exists():
        result["status"] = "MISSING_" + ANNOTATION_FILE
        return result

    # Load COCO JSON.
    try:
        with open(
            annotation_path,
            "r",
            encoding="utf-8"
        ) as f:
            coco = json.load(f)
    except Exception as e:
        result["status"] = f"INVALID_JSON: {e}"
        return result

    if not isinstance(coco, dict):
        result["status"] = "INVALID_COCO_ROOT"
        return result

    images = coco.get("images", [])
    annotations = coco.get("annotations", [])

    if not isinstance(images, list):
        result["status"] = "INVALID_IMAGES_LIST"
        return result

    if not isinstance(annotations, list):
        result["status"] = "INVALID_ANNOTATIONS_LIST"
        return result

    image_map = {
        image.get("id"): image
        for image in images
    }

    result["images"] = len(images)
    result["annotations_before"] = len(annotations)

    # Copy images.
    if not dry_run and copy_image_files:
        copied = copy_images(src_dir, dst_dir)
        print(f"  Images copied: {copied}")

    # Read actual dimensions.
    dimensions = {}

    for image in images:
        image_id = image.get("id")
        file_name = image.get("file_name", "")

        image_path = find_image(
            src_dir,
            file_name
        )

        if image_path is None:
            result["missing_images"] += 1

            result["missing_image_rows"].append({
                "split": split,
                "image_id": image_id,
                "file_name": file_name,
                "issue": "image_file_not_found"
            })

            result["issues"][
                "image_file_not_found"
            ] += 1

            continue

        width, height, error = get_image_dimensions(
            image_path
        )

        if error:
            result["image_read_errors"] += 1

            result["missing_image_rows"].append({
                "split": split,
                "image_id": image_id,
                "file_name": file_name,
                "issue": f"image_read_error: {error}"
            })

            result["issues"][
                "image_read_error"
            ] += 1

            continue

        dimensions[image_id] = (
            width,
            height
        )

        json_width = image.get("width")
        json_height = image.get("height")

        mismatch = ""

        try:
            if (
                int(json_width) != int(width)
                or int(json_height) != int(height)
            ):
                mismatch = (
                    f"json={json_width}x{json_height};"
                    f"actual={width}x{height}"
                )
        except Exception:
            mismatch = (
                f"json={json_width}x{json_height};"
                f"actual={width}x{height}"
            )

        if mismatch:
            result["issues"][
                "dimension_mismatch"
            ] += 1

        result["dimension_rows"].append({
            "split": split,
            "image_id": image_id,
            "file_name": file_name,
            "json_width": json_width,
            "json_height": json_height,
            "actual_width": width,
            "actual_height": height,
            "dimension_mismatch": mismatch
        })

        # Use actual image dimensions in cleaned COCO.
        if not dry_run:
            image["width"] = width
            image["height"] = height

    # Count annotations per image before cleanup.
    annotations_by_image = defaultdict(list)

    for annotation in annotations:
        annotations_by_image[
            annotation.get("image_id")
        ].append(annotation)

    cleaned_annotations = []

    for annotation in annotations:

        annotation_id = annotation.get("id")
        image_id = annotation.get("image_id")
        category_id = annotation.get("category_id")
        bbox = annotation.get("bbox")

        # Annotation references an image that doesn't exist
        # in the COCO images list.
        if image_id not in image_map:

            result["annotations_removed"] += 1

            result["issues"][
                "annotation_references_missing_image"
            ] += 1

            result["changes"].append({
                "split": split,
                "image_id": image_id,
                "image": "",
                "annotation_id": annotation_id,
                "category_id": category_id,
                "issue": "annotation_references_missing_image",
                "original_bbox": json.dumps(
                    bbox,
                    ensure_ascii=False
                ),
                "corrected_bbox": "",
                "action": "removed"
            })

            continue

        file_name = image_map[
            image_id
        ].get("file_name", "")

        # Cannot validate without actual image dimensions.
        if image_id not in dimensions:

            cleaned_annotations.append(
                annotation
            )

            result["issues"][
                "image_dimension_unavailable"
            ] += 1

            continue

        img_w, img_h = dimensions[
            image_id
        ]

        corrected_bbox, issues, action, original_bbox = clean_bbox(
            bbox,
            img_w,
            img_h
        )

        for issue in issues:
            result["issues"][issue] += 1

        if action == "removed":

            result["annotations_removed"] += 1

            result["changes"].append({
                "split": split,
                "image_id": image_id,
                "image": file_name,
                "annotation_id": annotation_id,
                "category_id": category_id,
                "issue": ";".join(issues),
                "original_bbox": json.dumps(
                    original_bbox,
                    ensure_ascii=False
                ),
                "corrected_bbox": "",
                "action": "removed"
            })

            continue

        if action == "clipped":

            result["annotations_clipped"] += 1

            annotation["bbox"] = corrected_bbox

            # COCO area = width * height.
            annotation["area"] = round(
                float(
                    corrected_bbox[2]
                    * corrected_bbox[3]
                ),
                6
            )

            result["changes"].append({
                "split": split,
                "image_id": image_id,
                "image": file_name,
                "annotation_id": annotation_id,
                "category_id": category_id,
                "issue": ";".join(issues),
                "original_bbox": json.dumps(
                    original_bbox,
                    ensure_ascii=False
                ),
                "corrected_bbox": json.dumps(
                    corrected_bbox,
                    ensure_ascii=False
                ),
                "action": "clipped"
            })

        cleaned_annotations.append(
            annotation
        )

    # Replace annotation list.
    coco["annotations"] = cleaned_annotations

    result["annotations_after"] = len(
        cleaned_annotations
    )

    # Find images that have no remaining annotation.
    remaining_counts = Counter(
        annotation.get("image_id")
        for annotation in cleaned_annotations
    )

    for image in images:

        image_id = image.get("id")

        if remaining_counts.get(image_id, 0) == 0:

            result["empty_label_images"] += 1

            result["issues"][
                "empty_labels"
            ] += 1

            result["empty_images"].append({
                "split": split,
                "image_id": image_id,
                "file_name": image.get(
                    "file_name",
                    ""
                ),
                "reason": "no_remaining_annotations"
            })

    # Write cleaned COCO.
    if not dry_run:

        dst_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        output_annotation = (
            dst_dir / ANNOTATION_FILE
        )

        with open(
            output_annotation,
            "w",
            encoding="utf-8"
        ) as f:
            json.dump(
                coco,
                f,
                indent=2,
                ensure_ascii=False
            )

    result["status"] = "OK"

    result["issues"] = dict(
        result["issues"]
    )

    return result


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Clean invalid COCO annotations "
            "for train/test/val/val_eval."
        )
    )

    parser.add_argument(
        "--root",
        required=True,
        help=(
            "Original dataset root containing "
            "train/test/val/val_eval."
        )
    )

    parser.add_argument(
        "--out",
        required=True,
        help=(
            "Output directory where the cleaned "
            "dataset will be created."
        )
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Analyze only. Do not create the "
            "cleaned dataset."
        )
    )

    parser.add_argument(
        "--no-copy-images",
        action="store_true",
        help=(
            "Do not copy image files. "
            "By default images are copied."
        )
    )

    args = parser.parse_args()

    root = Path(
        args.root
    ).expanduser().resolve()

    output = Path(
        args.out
    ).expanduser().resolve()

    if not root.exists():
        print(
            f"ERROR: Dataset does not exist: {root}"
        )
        sys.exit(1)

    if root == output:
        print(
            "ERROR: --out must be different "
            "from --root."
        )
        sys.exit(1)

    if not args.dry_run:
        output.mkdir(
            parents=True,
            exist_ok=True
        )

    report_dir = (
        output / "cleanup_reports"
    )

    report_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    print("=" * 80)
    print("COCO ANNOTATION CLEANUP")
    print("=" * 80)
    print(f"Input : {root}")
    print(f"Output: {output}")
    print()

    split_results = []

    all_changes = []
    all_empty = []
    all_missing = []
    all_dimensions = []

    for split in SPLITS:

        print("-" * 80)
        print(f"Processing {split}")

        result = process_split(
            src_root=root,
            dst_root=output,
            split=split,
            dry_run=args.dry_run,
            copy_image_files=(
                not args.no_copy_images
            )
        )

        split_results.append(
            result
        )

        all_changes.extend(
            result["changes"]
        )

        all_empty.extend(
            result["empty_images"]
        )

        all_missing.extend(
            result["missing_image_rows"]
        )

        all_dimensions.extend(
            result["dimension_rows"]
        )

        print(
            f"  Status             : "
            f"{result['status']}"
        )

        print(
            f"  Images             : "
            f"{result['images']}"
        )

        print(
            f"  Annotations before : "
            f"{result['annotations_before']}"
        )

        print(
            f"  Annotations after  : "
            f"{result['annotations_after']}"
        )

        print(
            f"  Clipped            : "
            f"{result['annotations_clipped']}"
        )

        print(
            f"  Removed            : "
            f"{result['annotations_removed']}"
        )

        print(
            f"  Empty-label images : "
            f"{result['empty_label_images']}"
        )

        if result["issues"]:

            print("  Issues:")

            for issue, count in sorted(
                result["issues"].items()
            ):
                print(
                    f"    {issue}: {count}"
                )

    # Overall totals.
    totals = {
        "images": sum(
            r["images"]
            for r in split_results
        ),
        "annotations_before": sum(
            r["annotations_before"]
            for r in split_results
        ),
        "annotations_after": sum(
            r["annotations_after"]
            for r in split_results
        ),
        "annotations_clipped": sum(
            r["annotations_clipped"]
            for r in split_results
        ),
        "annotations_removed": sum(
            r["annotations_removed"]
            for r in split_results
        ),
        "empty_label_images": sum(
            r["empty_label_images"]
            for r in split_results
        ),
        "missing_images": sum(
            r["missing_images"]
            for r in split_results
        ),
        "image_read_errors": sum(
            r["image_read_errors"]
            for r in split_results
        )
    }

    overall_issues = Counter()

    for result in split_results:
        overall_issues.update(
            result["issues"]
        )

    # Reports.
    write_csv(
        report_dir / "annotation_changes.csv",
        all_changes,
        [
            "split",
            "image_id",
            "image",
            "annotation_id",
            "category_id",
            "issue",
            "original_bbox",
            "corrected_bbox",
            "action"
        ]
    )

    write_csv(
        report_dir / "empty_label_images.csv",
        all_empty,
        [
            "split",
            "image_id",
            "file_name",
            "reason"
        ]
    )

    write_csv(
        report_dir / "missing_or_unreadable_images.csv",
        all_missing,
        [
            "split",
            "image_id",
            "file_name",
            "issue"
        ]
    )

    write_csv(
        report_dir / "image_dimensions.csv",
        all_dimensions,
        [
            "split",
            "image_id",
            "file_name",
            "json_width",
            "json_height",
            "actual_width",
            "actual_height",
            "dimension_mismatch"
        ]
    )

    write_csv(
        report_dir / "split_summary.csv",
        [
            {
                "split": r["split"],
                "status": r["status"],
                "images": r["images"],
                "annotations_before":
                    r["annotations_before"],
                "annotations_after":
                    r["annotations_after"],
                "annotations_clipped":
                    r["annotations_clipped"],
                "annotations_removed":
                    r["annotations_removed"],
                "empty_label_images":
                    r["empty_label_images"],
                "missing_images":
                    r["missing_images"],
                "image_read_errors":
                    r["image_read_errors"]
            }
            for r in split_results
        ],
        [
            "split",
            "status",
            "images",
            "annotations_before",
            "annotations_after",
            "annotations_clipped",
            "annotations_removed",
            "empty_label_images",
            "missing_images",
            "image_read_errors"
        ]
    )

    summary = {
        "input_root": str(root),
        "output_root": str(output),
        "dry_run": args.dry_run,
        "splits": SPLITS,
        "totals": totals,
        "issue_counts": dict(
            overall_issues
        ),
        "per_split": [
            {
                "split": r["split"],
                "status": r["status"],
                "images": r["images"],
                "annotations_before":
                    r["annotations_before"],
                "annotations_after":
                    r["annotations_after"],
                "annotations_clipped":
                    r["annotations_clipped"],
                "annotations_removed":
                    r["annotations_removed"],
                "empty_label_images":
                    r["empty_label_images"],
                "missing_images":
                    r["missing_images"],
                "image_read_errors":
                    r["image_read_errors"],
                "issues": r["issues"]
            }
            for r in split_results
        ]
    }

    with open(
        report_dir / "cleanup_summary.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            summary,
            f,
            indent=2,
            ensure_ascii=False
        )

    print()
    print("=" * 80)
    print("OVERALL SUMMARY")
    print("=" * 80)

    print(
        f"Images                  : "
        f"{totals['images']}"
    )

    print(
        f"Annotations before      : "
        f"{totals['annotations_before']}"
    )

    print(
        f"Annotations after       : "
        f"{totals['annotations_after']}"
    )

    print(
        f"Annotations clipped     : "
        f"{totals['annotations_clipped']}"
    )

    print(
        f"Annotations removed     : "
        f"{totals['annotations_removed']}"
    )

    print(
        f"Empty-label images      : "
        f"{totals['empty_label_images']}"
    )

    print(
        f"Missing image files     : "
        f"{totals['missing_images']}"
    )

    print(
        f"Image read errors       : "
        f"{totals['image_read_errors']}"
    )

    print()
    print("Issue counts:")

    for issue, count in sorted(
        overall_issues.items()
    ):
        print(
            f"  {issue}: {count}"
        )

    if args.dry_run:
        print()
        print(
            "DRY RUN: no cleaned dataset "
            "was created."
        )
    else:
        print()
        print(
            f"Cleaned dataset: {output}"
        )

    print(
        f"Reports: {report_dir}"
    )

    print()
    print("Done.")


if __name__ == "__main__":
    main()
