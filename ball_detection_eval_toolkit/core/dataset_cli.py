"""
Shared argparse wiring so all eval_*.py scripts expose the same three
annotation-format options consistently, and build the right dataset object.
"""

import argparse

from core.dataset import build_dataset, BaseBallDataset


def add_dataset_args(parser: argparse.ArgumentParser):
    parser.add_argument("--ann-format", required=True, choices=["client", "coco", "yolo"],
                         help="Ground-truth annotation format to read.")

    # client (Nikon) format
    parser.add_argument("--data-root", default=None,
                         help="[client format] root dir containing <sport>/imgs, <sport>/labels")
    parser.add_argument("--ignore-dirs", nargs="*", default=None,
                         help="[client format] subdirectory names to skip (default: 'other')")

    # coco format
    parser.add_argument("--coco-ann", default=None,
                         help="[coco format] path to COCO instances json")
    parser.add_argument("--coco-image-dir", default=None,
                         help="[coco format] directory containing the images referenced in the json")
    parser.add_argument("--coco-category", default=None,
                         help="[coco format] category name to treat as 'ball' (auto-detected if omitted)")

    # yolo format
    parser.add_argument("--yolo-images", default=None,
                         help="[yolo format] images directory")
    parser.add_argument("--yolo-labels", default=None,
                         help="[yolo format] labels directory (mirrors images dir structure, .txt per image)")
    parser.add_argument("--yolo-ball-class-id", type=int, default=0,
                         help="[yolo format] class index representing 'ball' (default 0)")
    parser.add_argument("--yolo-sport", default=None,
                         help="[yolo format] force a single sport label for all images "
                              "(if the YOLO set isn't split into sport subfolders)")

    parser.add_argument("--sports", nargs="*", default=None,
                         help="Optional sport filter applied after loading, e.g. basketball soccer")


def build_dataset_from_args(args) -> BaseBallDataset:
    if args.ann_format == "client":
        if not args.data_root:
            raise ValueError("--data-root is required when --ann-format client")
        return build_dataset("client", root=args.data_root,
                              sport_filter=args.sports, ignore_dirs=args.ignore_dirs)

    elif args.ann_format == "coco":
        if not args.coco_ann or not args.coco_image_dir:
            raise ValueError("--coco-ann and --coco-image-dir are required when --ann-format coco")
        return build_dataset("coco", ann_file=args.coco_ann, image_dir=args.coco_image_dir,
                              category_name=args.coco_category, sport_filter=args.sports)

    elif args.ann_format == "yolo":
        if not args.yolo_images or not args.yolo_labels:
            raise ValueError("--yolo-images and --yolo-labels are required when --ann-format yolo")
        return build_dataset("yolo", images_dir=args.yolo_images, labels_dir=args.yolo_labels,
                              ball_class_id=args.yolo_ball_class_id,
                              sport_override=args.yolo_sport, sport_filter=args.sports)

    else:
        raise ValueError(f"Unknown --ann-format {args.ann_format!r}")
