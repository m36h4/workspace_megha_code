"""PicoDet evaluation beyond COCOeval: TP/FP/FN counts, a multi-IoU sweep, recall@FPPI,
and annotated FP/FN sample images -- all computed directly (greedy IoU matching), not
through pycocotools, so you can see exactly what's being counted.

A note on "TN": true negatives aren't a well-defined concept in object detection (the
background is an infinite space of non-boxes, so there's no denominator to count against).
This script does NOT report a TN number. What it reports instead:
  TP = a prediction that matches an unmatched GT box of the same class at IoU >= threshold
  FP = a prediction that matches no such GT box
  FN = a GT box that no prediction matched
Precision = TP/(TP+FP), Recall = TP/(TP+FN) -- standard detection convention.

Matching is greedy, per class, per image, processing all predictions in DESCENDING
confidence order and giving each one first claim on the best remaining unmatched GT box
above the IoU threshold -- the same convention COCOeval uses internally.

Usage (ESNet checkpoint):
    python picodet_eval.py --data dataset.yaml --weights runs/train/esnet_exp/weights/best.pt \
        --arch esnet --act leakyrelu --gate sigmoid --imgsz 320 448 \
        --iou 0.5 --conf 0.25 --fppi-iou 0.4 --fppi-target 0.05 --vis 12

Usage (LCNet checkpoint):
    python picodet_eval.py --data dataset.yaml --weights runs/train/lcnet_exp/weights/best.pt \
        --arch lcnet --act silu --gate sigmoid --imgsz 320 448 \
        --iou 0.5 --conf 0.25 --vis 12

Requires picodet_letterbox.py and (for --arch lcnet) picodet_lcnet_backbone.py +
picodet_activation.py in the same folder -- same as your training scripts.
"""
import argparse
import json
import os
from collections import defaultdict

import cv2
import numpy as np

import picodet_letterbox as pl  # letterbox pre/postprocessing -- import before building the model


def load_model(args):
    from picodet_activation import set_activation
    set_activation(args.act, gate=args.gate)

    if args.arch == "esnet":
        from libreyolo.models.picodet.model import LibrePICODET
        return LibrePICODET(args.weights, size=args.size)
    else:
        from picodet_lcnet_backbone import PicoDetLCNet
        return PicoDetLCNet(args.weights, size=args.size, lcnet_scale=args.lcnet_scale,
                            act=args.act, gate=args.gate)


def load_ground_truth(data_yaml: str, split: str):
    """Returns (image_id -> file_path, image_id -> [x1,y1,x2,y2,cls], class_names)."""
    from pycocotools.coco import COCO
    from libreyolo.data import load_data_config

    cfg = load_data_config(data_yaml)
    ann_file = cfg[f"{split}_annotation_file"]
    img_dir = cfg[split]
    coco = COCO(ann_file)

    cat_ids = sorted(coco.getCatIds())
    cat_id_to_idx = {c: i for i, c in enumerate(cat_ids)}
    names = cfg.get("names")
    class_names = [names[i] for i in range(len(names))] if isinstance(names, dict) else (
        names or [coco.loadCats([c])[0]["name"] for c in cat_ids])

    images, gts = {}, {}
    for img_id in coco.getImgIds():
        info = coco.loadImgs(img_id)[0]
        images[img_id] = os.path.join(img_dir, info["file_name"])
        rows = []
        for a in coco.loadAnns(coco.getAnnIds(imgIds=img_id)):
            x, y, w, h = a["bbox"]
            rows.append([x, y, x + w, y + h, cat_id_to_idx[a["category_id"]]])
        gts[img_id] = np.array(rows, dtype=np.float32) if rows else np.zeros((0, 5), dtype=np.float32)
    return images, gts, class_names


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    tl = np.maximum(a[:, None, :2], b[:, :2])
    br = np.minimum(a[:, None, 2:4], b[:, 2:4])
    inter = np.prod(np.clip(br - tl, 0, None), axis=2)
    area_a = np.prod(a[:, 2:4] - a[:, :2], axis=1)
    area_b = np.prod(b[:, 2:4] - b[:, :2], axis=1)
    return inter / (area_a[:, None] + area_b - inter + 1e-10)


def greedy_match(dets: np.ndarray, gt: np.ndarray, iou_thr: float):
    """dets: [x1,y1,x2,y2,score,cls] sorted by score DESC. gt: [x1,y1,x2,y2,cls].
    Returns a boolean array (len(dets),): True = TP, False = FP; and which GT indices
    were matched (for FN = the complement)."""
    matched_gt = np.zeros(len(gt), dtype=bool)
    is_tp = np.zeros(len(dets), dtype=bool)
    for i, d in enumerate(dets):
        cls = d[5]
        cand = np.where((gt[:, 4] == cls) & (~matched_gt))[0] if len(gt) else np.array([], dtype=int)
        if len(cand) == 0:
            continue
        ious = iou_matrix(d[None, :4], gt[cand, :4])[0]
        best = np.argmax(ious)
        if ious[best] >= iou_thr:
            matched_gt[cand[best]] = True
            is_tp[i] = True
    return is_tp, matched_gt


def run_inference(model, images: dict, imgsz, conf_floor: float, iou_thres: float, max_det: int):
    """Runs letterbox-aware inference on every image. Returns per-image-id predictions
    as [x1,y1,x2,y2,score,cls] arrays, all in ORIGINAL image pixel coordinates."""
    preds = {}
    for img_id, path in images.items():
        res = pl.letterbox_predict(model, path, target_hw=imgsz, conf_thres=conf_floor,
                                   iou_thres=iou_thres, max_det=max_det)
        if res["num_detections"]:
            boxes = np.asarray(res["boxes"], dtype=np.float32)
            scores = np.asarray(res["scores"], dtype=np.float32).reshape(-1, 1)
            classes = np.asarray(res["classes"], dtype=np.float32).reshape(-1, 1)
            preds[img_id] = np.concatenate([boxes, scores, classes], axis=1)
        else:
            preds[img_id] = np.zeros((0, 6), dtype=np.float32)
    return preds


def report_at_threshold(preds: dict, gts: dict, iou_thr: float, conf_thr: float, class_names):
    """TP/FP/FN overall and per class, at one (iou, conf) operating point."""
    per_class = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    total = {"tp": 0, "fp": 0, "fn": 0}
    for img_id, gt in gts.items():
        d = preds[img_id]
        d = d[d[:, 4] >= conf_thr]
        d = d[np.argsort(-d[:, 4])]
        is_tp, matched_gt = greedy_match(d, gt, iou_thr)
        classes_here = np.concatenate([d[:, 5], gt[:, 4]]) if (len(d) or len(gt)) else np.array([])
        for cls in np.unique(classes_here):
            cls_tp = int(is_tp[d[:, 5] == cls].sum()) if len(d) else 0
            cls_fp = int((~is_tp[d[:, 5] == cls]).sum()) if len(d) else 0
            cls_fn = int((~matched_gt[gt[:, 4] == cls]).sum()) if len(gt) else 0
            per_class[int(cls)]["tp"] += cls_tp
            per_class[int(cls)]["fp"] += cls_fp
            per_class[int(cls)]["fn"] += cls_fn
        total["tp"] += int(is_tp.sum())
        total["fp"] += int((~is_tp).sum())
        total["fn"] += int((~matched_gt).sum())

    def prec_rec(c):
        pr = c["tp"] / (c["tp"] + c["fp"] + 1e-10)
        rc = c["tp"] / (c["tp"] + c["fn"] + 1e-10)
        return pr, rc

    print(f"\n=== TP/FP/FN at IoU={iou_thr}, conf>={conf_thr} ===")
    pr, rc = prec_rec(total)
    print(f"  OVERALL: TP={total['tp']} FP={total['fp']} FN={total['fn']}  "
          f"precision={pr:.4f} recall={rc:.4f}")
    for cls in sorted(per_class):
        name = class_names[cls] if cls < len(class_names) else str(cls)
        pr, rc = prec_rec(per_class[cls])
        c = per_class[cls]
        print(f"  {name:>15s}: TP={c['tp']:4d} FP={c['fp']:4d} FN={c['fn']:4d}  "
              f"precision={pr:.4f} recall={rc:.4f}")
    return total, dict(per_class)


def report_at_threshold_quiet(preds, gts, iou_thr, conf_thr):
    total = {"tp": 0, "fp": 0, "fn": 0}
    for img_id, gt in gts.items():
        d = preds[img_id]
        d = d[d[:, 4] >= conf_thr]
        d = d[np.argsort(-d[:, 4])]
        is_tp, matched_gt = greedy_match(d, gt, iou_thr)
        total["tp"] += int(is_tp.sum())
        total["fp"] += int((~is_tp).sum())
        total["fn"] += int((~matched_gt).sum())
    return total, None


def iou_sweep(preds, gts, ious, conf_thr):
    print(f"\n=== Multi-IoU sweep at conf>={conf_thr} ===")
    print(f"{'IoU':>6s} {'TP':>6s} {'FP':>6s} {'FN':>6s} {'Precision':>10s} {'Recall':>8s}")
    rows = []
    for iou_thr in ious:
        total, _ = report_at_threshold_quiet(preds, gts, iou_thr, conf_thr)
        pr = total["tp"] / (total["tp"] + total["fp"] + 1e-10)
        rc = total["tp"] / (total["tp"] + total["fn"] + 1e-10)
        print(f"{iou_thr:6.2f} {total['tp']:6d} {total['fp']:6d} {total['fn']:6d} {pr:10.4f} {rc:8.4f}")
        rows.append({"iou": iou_thr, **total, "precision": pr, "recall": rc})
    return rows


def recall_at_fppi(preds: dict, gts: dict, iou_thr: float, target_fppi: float):
    """One pass over ALL detections sorted by score descending (standard PR-curve sweep):
    cumulative TP/FP as a function of score threshold IS the same as re-matching at every
    threshold, since higher-score detections always get first claim on a GT box regardless
    of what threshold is later applied. Interpolates recall at the target FPPI."""
    num_images = len(gts)
    total_gt = sum(len(g) for g in gts.values())

    all_dets = []  # (score, img_id, box_idx)
    per_image_dets = {}
    for img_id, d in preds.items():
        order = np.argsort(-d[:, 4])
        per_image_dets[img_id] = d[order]
        for i, score in enumerate(d[order, 4]):
            all_dets.append((score, img_id, i))
    all_dets.sort(key=lambda x: -x[0])

    matched_gt = {img_id: np.zeros(len(gts[img_id]), dtype=bool) for img_id in gts}
    cum_tp, cum_fp = 0, 0
    points = []  # (fppi, recall, score) after each detection, in descending score order
    for score, img_id, i in all_dets:
        d = per_image_dets[img_id][i]
        gt = gts[img_id]
        cls = d[5]
        cand = np.where((gt[:, 4] == cls) & (~matched_gt[img_id]))[0] if len(gt) else np.array([], dtype=int)
        is_tp = False
        if len(cand) > 0:
            ious = iou_matrix(d[None, :4], gt[cand, :4])[0]
            best = np.argmax(ious)
            if ious[best] >= iou_thr:
                matched_gt[img_id][cand[best]] = True
                is_tp = True
        if is_tp:
            cum_tp += 1
        else:
            cum_fp += 1
        points.append((cum_fp / num_images, cum_tp / (total_gt + 1e-10), score))

    if not points:
        print(f"\n=== Recall @ FPPI={target_fppi} (IoU={iou_thr}) ===\n  no detections at all")
        return 0.0, None
    fppis = np.array([p[0] for p in points])
    recalls = np.array([p[1] for p in points])
    idx = np.searchsorted(fppis, target_fppi)
    if idx == 0:
        recall, thr = recalls[0], points[0][2]
    elif idx >= len(fppis):
        recall, thr = recalls[-1], points[-1][2]
    else:
        f0, f1 = fppis[idx - 1], fppis[idx]
        r0, r1 = recalls[idx - 1], recalls[idx]
        frac = 0.0 if f1 == f0 else (target_fppi - f0) / (f1 - f0)
        recall = r0 + frac * (r1 - r0)
        thr = points[idx][2]
    print(f"\n=== Recall @ FPPI={target_fppi} (IoU={iou_thr}) ===")
    print(f"  recall={recall:.4f} at an operating confidence threshold near {thr:.4f}")
    print(f"  (total GT={total_gt}, images={num_images})")
    return recall, thr


def denorm_and_draw(path, gt, dets, is_tp, matched_gt, class_names, out_path):
    img = cv2.imread(path)
    if len(gt):
        for x1, y1, x2, y2, cls in gt[~matched_gt]:
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 165, 255), 2)  # FN: orange
            cv2.putText(img, f"FN:{class_names[int(cls)]}", (int(x1), max(0, int(y1) - 5)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1, cv2.LINE_AA)
    for d, tp in zip(dets, is_tp):
        x1, y1, x2, y2, score, cls = d
        color = (0, 255, 0) if tp else (0, 0, 255)  # TP: green, FP: red
        label = ("TP" if tp else "FP") + f":{class_names[int(cls)]}:{score:.2f}"
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        cv2.putText(img, label, (int(x1), max(0, int(y1) - 5)),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    cv2.imwrite(out_path, img)


def save_fp_fn_samples(images, preds, gts, iou_thr, conf_thr, class_names, out_dir, n):
    os.makedirs(out_dir, exist_ok=True)
    scored = []
    for img_id, gt in gts.items():
        d = preds[img_id]
        d = d[d[:, 4] >= conf_thr]
        d = d[np.argsort(-d[:, 4])]
        is_tp, matched_gt = greedy_match(d, gt, iou_thr)
        n_bad = int((~is_tp).sum() + (~matched_gt).sum())
        scored.append((n_bad, img_id, d, is_tp, matched_gt))
    scored.sort(key=lambda x: -x[0])  # worst offenders first

    for n_bad, img_id, d, is_tp, matched_gt in scored[:n]:
        out_path = os.path.join(out_dir, f"img{img_id}_fp{int((~is_tp).sum())}_fn{int((~matched_gt).sum())}.png")
        denorm_and_draw(images[img_id], gts[img_id], d, is_tp, matched_gt, class_names, out_path)
    print(f"\nSaved {min(n, len(scored))} FP/FN sample images to {out_dir}/ "
          f"(green=TP, red=FP, orange=FN; worst images first)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--split", default="val")
    p.add_argument("--weights", required=True)
    p.add_argument("--arch", required=True, choices=["esnet", "lcnet"])
    p.add_argument("--size", default="s", choices=["s", "m", "l"])
    p.add_argument("--lcnet-scale", type=float, default=0.75)
    p.add_argument("--act", default="hswish")
    p.add_argument("--gate", default="default")
    p.add_argument("--imgsz", type=int, nargs=2, default=[320, 448], metavar=("H", "W"))
    p.add_argument("--iou", type=float, default=0.5, help="IoU threshold for the main TP/FP/FN report")
    p.add_argument("--conf", type=float, default=0.25, help="confidence threshold for the main report")
    p.add_argument("--iou-sweep", type=float, nargs="+", default=[0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.9])
    p.add_argument("--fppi-iou", type=float, default=0.4)
    p.add_argument("--fppi-target", type=float, default=0.05)
    p.add_argument("--conf-floor", type=float, default=0.001, help="low conf floor used to build the full ranked list for the FPPI sweep")
    p.add_argument("--nms-iou", type=float, default=0.45, help="NMS IoU used inside inference, not the eval-matching IoU")
    p.add_argument("--max-det", type=int, default=300)
    p.add_argument("--vis", type=int, default=10, help="number of FP/FN sample images to save (0 = skip)")
    p.add_argument("--out", default="eval_out")
    args = p.parse_args()

    model = load_model(args)
    images, gts, class_names = load_ground_truth(args.data, args.split)
    print(f"{args.split}: {len(images)} images, {sum(len(g) for g in gts.values())} GT boxes, "
          f"classes={class_names}")

    preds = run_inference(model, images, tuple(args.imgsz), args.conf_floor, args.nms_iou, args.max_det)

    report_at_threshold(preds, gts, args.iou, args.conf, class_names)
    sweep_rows = iou_sweep(preds, gts, args.iou_sweep, args.conf)
    recall_at_fppi(preds, gts, args.fppi_iou, args.fppi_target)

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "iou_sweep.json"), "w") as f:
        json.dump(sweep_rows, f, indent=2)

    if args.vis > 0:
        save_fp_fn_samples(images, preds, gts, args.iou, args.conf, class_names,
                          os.path.join(args.out, "fp_fn_samples"), args.vis)


if __name__ == "__main__":
    main()
