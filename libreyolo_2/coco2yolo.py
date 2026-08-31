"""
Convert standard COCO-format JSON to YOLO-format .txt labels.

COCO bbox format is well-defined: [x_min, y_min, width, height], absolute
pixels, top-left origin -- no ambiguity, unlike custom formats.

Usage:
  python3 coco_to_yolo.py \
      --coco-json /path/to/train.json \
      --out-dir /path/to/dataset/labels/train \
      --classes-out /path/to/dataset/classes.txt

Run once per split (train.json and val.json separately). Use
--classes-out only on the FIRST call (e.g. train.json) and pass
--classes-in on subsequent calls (val.json) so class ids stay
consistent across splits.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coco-json", required=True, help="Path to COCO-format JSON (train.json or val.json)")
    ap.add_argument("--out-dir", required=True, help="Output directory for YOLO .txt label files")
    ap.add_argument("--classes-out", default=None, help="Write class list here (use for the first split)")
    ap.add_argument("--classes-in", default=None, help="Read existing class list from here (use for subsequent splits, to keep ids consistent)")
    args = ap.parse_args()

    if not args.classes_out and not args.classes_in:
        raise SystemExit("Pass either --classes-out (first split) or --classes-in (later splits)")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.coco_json) as f:
        coco = json.load(f)

    # Build category_id -> class_id mapping
    categories = sorted(coco["categories"], key=lambda c: c["id"])
    if args.classes_in:
        with open(args.classes_in) as f:
            class_names = [line.strip() for line in f if line.strip()]
        name_to_class_id = {name: i for i, name in enumerate(class_names)}
        catid_to_classid = {}
        for cat in categories:
            if cat["name"] not in name_to_class_id:
                raise SystemExit(
                    f"Category '{cat['name']}' in {args.coco_json} not found in {args.classes_in}. "
                    f"Class sets must match across splits."
                )
            catid_to_classid[cat["id"]] = name_to_class_id[cat["name"]]
    else:
        class_names = [cat["name"] for cat in categories]
        catid_to_classid = {cat["id"]: i for i, cat in enumerate(categories)}
        with open(args.classes_out, "w") as f:
            for name in class_names:
                f.write(name + "\n")
        print(f"Found {len(class_names)} classes: {class_names}")

    # image_id -> image info
    images_by_id = {img["id"]: img for img in coco["images"]}

    # image_id -> list of annotations
    anns_by_image = defaultdict(list)
    for ann in coco["annotations"]:
        if ann.get("iscrowd", 0) == 1:
            continue  # skip crowd regions, standard practice
        anns_by_image[ann["image_id"]].append(ann)

    n_boxes = 0
    n_skipped = 0
    n_images_no_ann = 0

    for img_id, img_info in images_by_id.items():
        img_w = img_info["width"]
        img_h = img_info["height"]
        stem = Path(img_info["file_name"]).stem

        lines = []
        for ann in anns_by_image.get(img_id, []):
            x, y, w, h = ann["bbox"]  # COCO: top-left x, y, width, height
            if w <= 0 or h <= 0:
                n_skipped += 1
                continue

            cls_id = catid_to_classid[ann["category_id"]]
            x_center = (x + w / 2.0) / img_w
            y_center = (y + h / 2.0) / img_h
            w_norm = w / img_w
            h_norm = h / img_h

            lines.append(f"{cls_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")
            n_boxes += 1

        if not lines:
            n_images_no_ann += 1

        out_path = out_dir / f"{stem}.txt"
        with open(out_path, "w") as f:
            f.write("\n".join(lines))  # empty file = valid background image

    print(f"Converted {len(images_by_id)} images -> {n_boxes} boxes written, {n_skipped} degenerate boxes skipped")
    print(f"{n_images_no_ann} images had zero valid annotations (written as empty .txt)")


if __name__ == "__main__":
    main()
