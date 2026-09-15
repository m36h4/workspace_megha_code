#!/usr/bin/env python3
"""
Evaluate a NanoDet-Plus (.pt) ball-detection model against ground truth in
ANY of three annotation formats: client (Nikon), coco, or yolo.

*** REQUIRES models/nanodet_backend.py to be completed first *** -- see that
file's docstring. Once the NanoDet-Plus architecture code + training config
are available, this script needs NO changes; it already handles all dataset
formats, metrics, and reporting identically to eval_onnx.py.

Examples:

    python scripts/eval_nanodet.py --checkpoint model.pt --config nanodet-plus-m.yml \
        --ann-format client --data-root /path/to/basic_eval_caseA \
        --input-w 480 --input-h 320 --out-dir results/nanodet_client

    python scripts/eval_nanodet.py --checkpoint model.pt --config nanodet-plus-m.yml \
        --ann-format coco --coco-ann /path/to/instances.json --coco-image-dir /path/to/images \
        --out-dir results/nanodet_coco

    python scripts/eval_nanodet.py --checkpoint model.pt --config nanodet-plus-m.yml \
        --ann-format yolo --yolo-images /path/to/images --yolo-labels /path/to/labels \
        --out-dir results/nanodet_yolo
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.dataset import load_image_rgb
from core.dataset_cli import add_dataset_args, build_dataset_from_args
from core.preprocess import xywh_to_xyxy
from core.metrics import PerImagePrediction, evaluate, collect_fp_fn_samples
from core.report import write_markdown_report, plot_fppi_recall_curve, draw_fp_fn_overlays, compute_gmacs_torchinfo
from models.nanodet_backend import NanoDetBallDetector


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True, help="Path to NanoDet-Plus .pt checkpoint")
    p.add_argument("--config", required=True, help="Path to the training .yml config (required for decoding)")
    add_dataset_args(p)

    p.add_argument("--input-w", type=int, default=480)
    p.add_argument("--input-h", type=int, default=320)
    p.add_argument("--score-thresh", type=float, default=0.01)
    p.add_argument("--nms-iou", type=float, default=0.5)
    p.add_argument("--iou-thresh", type=float, default=0.4, help="Matching IoU threshold for TP (client spec: 0.4)")
    p.add_argument("--target-fppi", type=float, default=0.05, help="Client spec: 0.05")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--dump-fp-fn", action="store_true")
    p.add_argument("--compute-gmacs", action="store_true",
                   help="Also compute GMACs via torchinfo (report item 3) -- works for this backend "
                        "since it's a native PyTorch model")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading ground truth (format={args.ann_format}) ...")
    dataset = build_dataset_from_args(args)
    print("Dataset stats:", dataset.stats())
    if len(dataset) == 0:
        print("ERROR: no samples found. Check your annotation-format arguments.")
        sys.exit(1)

    print(f"Loading NanoDet-Plus checkpoint from {args.checkpoint} ...")
    detector = NanoDetBallDetector(
        args.checkpoint, args.config,
        input_size=(args.input_w, args.input_h),
        score_thresh=args.score_thresh, nms_iou_thresh=args.nms_iou,
        device=args.device,
    )

    records = []
    image_id_to_path = {}
    for i, sample in enumerate(dataset):
        image_id_to_path[sample.image_id] = sample.image_path
        img_rgb = load_image_rgb(sample.image_path)
        boxes_xyxy, scores = detector.predict(img_rgb)
        gt_xyxy = xywh_to_xyxy(sample.gt_boxes_xywh)

        records.append(PerImagePrediction(
            image_id=sample.image_id, sport=sample.sport,
            pred_boxes_xyxy=boxes_xyxy, pred_scores=scores,
            gt_boxes_xyxy=gt_xyxy,
        ))
        if (i + 1) % 200 == 0:
            print(f"  processed {i + 1}/{len(dataset)}")

    print("Running evaluation ...")
    result = evaluate(records, iou_thresh=args.iou_thresh, target_fppi=args.target_fppi)

    print(f"\nOverall Recall@FPPI={args.target_fppi}: {result['overall']['recall_at_target_fppi']:.4f}")
    for sport, res in result["per_sport"].items():
        print(f"  {sport}: {res['recall_at_target_fppi']:.4f}  (n_gt={res['n_gt']}, n_img={res['n_images']})")

    gmacs_info = None
    if args.compute_gmacs:
        try:
            gmacs_info = compute_gmacs_torchinfo(detector.model, input_size=(1, 3, args.input_h, args.input_w))
            print(f"GMACs: {gmacs_info['gmacs']:.4f}")
        except Exception as e:
            print(f"WARNING: GMACs computation failed: {e}")

    curve_png = None
    if result["overall"]["curve"]["fppi"]:
        curve_png = str(out_dir / "recall_vs_fppi.png")
        plot_fppi_recall_curve(result["overall"]["curve"], curve_png, args.target_fppi,
                                title=f"NanoDet-Plus — Recall vs FPPI ({args.ann_format} GT)")

    fp_fn_dir = None
    if args.dump_fp_fn and result["overall"]["operating_threshold"] is not None:
        fp_fn_data = collect_fp_fn_samples(records, threshold=result["overall"]["operating_threshold"],
                                            iou_thresh=args.iou_thresh, max_samples=30)
        fp_fn_dir = str(out_dir / "fp_fn_samples")
        draw_fp_fn_overlays(fp_fn_data, "", fp_fn_dir, image_id_to_path=lambda iid: image_id_to_path[iid])

    report_path, json_path = write_markdown_report(
        result, model_name=f"{Path(args.checkpoint).stem} [{args.ann_format}]",
        out_path=str(out_dir / "report.md"), gmacs_info=gmacs_info,
        recall_curve_png="recall_vs_fppi.png" if curve_png else None,
        fp_fn_dir="fp_fn_samples" if fp_fn_dir else None,
    )
    print(f"\nReport written to: {report_path}")
    print(f"Raw results JSON: {json_path}")


if __name__ == "__main__":
    main()
