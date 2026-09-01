"""
Symlink images referenced by a COCO json into a target images/<split> folder.
Run once per split.

Usage:
  python3 link_coco_images.py \
      --coco-json /path/to/train.json \
      --images-src /path/to/all_images \
      --images-dst /path/to/dataset/images/train
"""
import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coco-json", required=True)
    ap.add_argument("--images-src", required=True, help="Folder containing the actual image files")
    ap.add_argument("--images-dst", required=True, help="Target images/train or images/val folder")
    args = ap.parse_args()

    src = Path(args.images_src)
    dst = Path(args.images_dst)
    dst.mkdir(parents=True, exist_ok=True)

    with open(args.coco_json) as f:
        coco = json.load(f)

    n_ok, n_missing = 0, 0
    for img in coco["images"]:
        fname = Path(img["file_name"]).name
        src_path = src / fname
        if not src_path.exists():
            n_missing += 1
            continue
        link_path = dst / fname
        if not link_path.exists():
            link_path.symlink_to(src_path.resolve())
        n_ok += 1

    print(f"Linked {n_ok} images into {dst}")
    if n_missing:
        print(f"WARNING: {n_missing} images referenced in JSON were not found in {src}")


if __name__ == "__main__":
    main()
