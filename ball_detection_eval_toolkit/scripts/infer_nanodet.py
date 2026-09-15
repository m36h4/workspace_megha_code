#!/usr/bin/env python3
"""
Run NanoDet-Plus (.pt) inference on a single image or directory, drawing
predicted ball boxes for visual sanity-checking. Optional GT overlay in any
of the three annotation formats (client/coco/yolo), same as infer_onnx.py.

*** REQUIRES models/nanodet_backend.py to be completed first ***

Example:
    python scripts/infer_nanodet.py --checkpoint model.pt --config nanodet-plus-m.yml \
        --input path/to/img_or_dir --out-dir vis_output
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
from core.dataset import load_image_rgb
from core.dataset_cli import add_dataset_args, build_dataset_from_args
from core.preprocess import xywh_to_xyxy
from models.nanodet_backend import NanoDetBallDetector


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--input", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--input-w", type=int, default=480)
    p.add_argument("--input-h", type=int, default=320)
    p.add_argument("--score-thresh", type=float, default=0.3)
    p.add_argument("--nms-iou", type=float, default=0.5)
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])

    p.add_argument("--ann-format", default=None, choices=[None, "client", "coco", "yolo"])
    p.add_argument("--data-root", default=None)
    p.add_argument("--ignore-dirs", nargs="*", default=None)
    p.add_argument("--coco-ann", default=None)
    p.add_argument("--coco-image-dir", default=None)
    p.add_argument("--coco-category", default=None)
    p.add_argument("--yolo-images", default=None)
    p.add_argument("--yolo-labels", default=None)
    p.add_argument("--yolo-ball-class-id", type=int, default=0)
    p.add_argument("--yolo-sport", default=None)
    p.add_argument("--sports", nargs="*", default=None)
    return p.parse_args()


def build_gt_lookup(args):
    if args.ann_format is None:
        return {}
    dataset = build_dataset_from_args(args)
    lookup = {}
    for sample in dataset:
        lookup[str(Path(sample.image_path).resolve())] = xywh_to_xyxy(sample.gt_boxes_xywh)
    print(f"Loaded {len(lookup)} GT-labeled images for overlay (format={args.ann_format})")
    return lookup


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    gt_lookup = build_gt_lookup(args)

    detector = NanoDetBallDetector(
        args.checkpoint, args.config, input_size=(args.input_w, args.input_h),
        score_thresh=args.score_thresh, nms_iou_thresh=args.nms_iou, device=args.device,
    )

    input_path = Path(args.input)
    if input_path.is_dir():
        image_paths = sorted([p for p in input_path.rglob("*")
                               if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}])
    else:
        image_paths = [input_path]

    for img_path in image_paths:
        img_rgb = load_image_rgb(str(img_path))
        boxes, scores = detector.predict(img_rgb)

        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        gt_boxes = gt_lookup.get(str(img_path.resolve()), np.zeros((0, 4), dtype=np.float32))
        for gt in gt_boxes:
            x1, y1, x2, y2 = [int(v) for v in gt]
            cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img_bgr, "GT", (x1, max(0, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        for box, score in zip(boxes, scores):
            x1, y1, x2, y2 = [int(v) for v in box]
            cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(img_bgr, f"ball {score:.2f}", (x1, max(0, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        out_path = out_dir / f"{img_path.stem}_pred.jpg"
        cv2.imwrite(str(out_path), img_bgr)
        print(f"{img_path.name}: {len(boxes)} ball(s) detected -> {out_path}")


if __name__ == "__main__":
    main()
