#!/usr/bin/env python3
"""
eval_picodet.py
-----------------
Full evaluation for a LibreYOLO PicoDet checkpoint.

Outputs:
  1. Standard metrics via LibreYOLO's own validator (mAP50-95, mAP50, mAP75,
     precision, recall, etc.) -- printed + saved to metrics.json
  2. Loss curve plot, read from the training run's results.csv (the file
     LibreYOLO writes every epoch under <project>/<name>/results.csv).
  3. TP / FP / FN counts and Recall @ IoU=0.4, computed by matching raw
     detections against YOLO-format ground-truth labels.
  4. Recall @ FPPI=0.05 (recall at the confidence threshold where false
     positives per image == 0.05), via an FPPI-vs-recall sweep curve.

Requires: libreyolo, matplotlib, numpy, pillow, pyyaml

Usage:
    python eval_picodet.py \
        --weights last.pt \
        --data /path/to/dataset.yaml \
        --split val \
        --results-csv /path/to/train_run/results.csv \
        --out-dir ./eval_out
"""

import argparse
import importlib.util
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate a LibreYOLO PicoDet checkpoint")
    p.add_argument("--weights", required=True, help="Path to .pt checkpoint")
    p.add_argument("--data", required=True, help="Path to dataset.yaml (YOLO format)")
    p.add_argument("--split", default="val", help="Dataset split to evaluate on")
    p.add_argument("--imgsz", type=str, default=None,
                    help="Eval image size. Either a single int for square (e.g. 320) or "
                         "WxH for rectangular (e.g. 320x480 meaning width=320, height=480). "
                         "Defaults to the checkpoint's native size.")
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--conf", type=float, default=0.001, help="Confidence floor for raw detections")
    p.add_argument("--nms-iou", type=float, default=0.6, help="IoU threshold used for NMS")
    p.add_argument("--match-iou", type=float, default=0.4, help="IoU threshold for TP/FP/FN + recall")
    p.add_argument("--target-fppi", type=float, default=0.05, help="FPPI operating point")
    p.add_argument("--size", default=None, choices=["xs", "s", "m", "l"],
                    help="Override PicoDet size (xs/s/m/l). Default: auto-detect from checkpoint.")
    p.add_argument("--lcnet-script", default=None,
                    help="Path to ch_act_picodet_lcnet_backbone.py (or wherever your "
                         "PicoDetLCNet subclass lives). Required only for checkpoints "
                         "trained with the PP-LCNet backbone script -- auto-detected from "
                         "the checkpoint's key names, but the script location can't be "
                         "guessed, so pass it explicitly the first time you evaluate such "
                         "a checkpoint. The directory containing this file must also "
                         "contain picodet_activation.py (its sibling import).")
    p.add_argument("--lcnet-scale", type=float, default=0.75,
                    help="lcnet_scale the checkpoint was trained with (PicoDetLCNet only).")
    p.add_argument("--device", default="auto")
    p.add_argument("--results-csv", default=None,
                    help="Path to training results.csv for the loss curve. "
                         "If omitted, the script looks for one next to the weights file.")
    p.add_argument("--out-dir", default="./eval_out", help="Where to write plots/metrics/json")
    return p.parse_args()


def parse_imgsz(imgsz_arg):
    """Parses --imgsz into a (h, w) tuple, the order LibreYOLO's picodet
    preprocessing/val() expects. Accepts a single int ('320' -> square
    (320,320)) or 'WxH' / 'W,H' (e.g. '320x480' meaning width=320,
    height=480 -> returned as (480, 320))."""
    if imgsz_arg is None:
        return None
    s = str(imgsz_arg).lower().replace(",", "x")
    if "x" in s:
        w_str, h_str = s.split("x")
        w, h = int(w_str), int(h_str)
        return (h, w)
    v = int(s)
    return (v, v)


def resolve_device(device_arg):
    if device_arg == "auto":
        return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    return torch.device(device_arg)


def detect_backbone_kind(state_dict):
    """Inspects checkpoint key names to tell LibreYOLO's stock ESNet backbone
    apart from a custom PP-LCNet backbone (e.g. the PicoDetLCNet subclass in
    ch_act_picodet_lcnet_backbone.py). Returns "esnet", "lcnet", or "unknown"."""
    keys = list(state_dict.keys())
    has_esnet_pattern = any(".conv_dw_1." in k or ".conv_pw_2." in k for k in keys)
    has_lcnet_pattern = any(
        re.match(r"^backbone\.blocks[2-6]\.\d+\.(dw_conv|pw_conv)\.", k) for k in keys
    )
    if has_esnet_pattern and not has_lcnet_pattern:
        return "esnet"
    if has_lcnet_pattern and not has_esnet_pattern:
        return "lcnet"
    return "unknown"


def import_picodet_lcnet(script_path):
    """Dynamically imports PicoDetLCNet from a user-supplied script path.
    The script's sibling import (picodet_activation.py) must sit in the same
    directory -- this adds that directory to sys.path so the import resolves
    the same way it would if you ran the script directly."""
    script_path = Path(script_path).resolve()
    if not script_path.is_file():
        raise SystemExit(f"--lcnet-script not found: {script_path}")

    script_dir = str(script_path.parent)
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)

    spec = importlib.util.spec_from_file_location("_picodet_lcnet_backbone_module", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, "PicoDetLCNet"):
        raise SystemExit(
            f"{script_path} was imported but has no PicoDetLCNet class. "
            "Pass the correct --lcnet-script path."
        )
    return module.PicoDetLCNet


def load_model(weights_path, device, size_override=None, lcnet_script=None, lcnet_scale=0.75):
    ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt

    kind = detect_backbone_kind(state_dict)

    size = size_override or (ckpt.get("size") if isinstance(ckpt, dict) else None)
    nb_classes = ckpt.get("nc") if isinstance(ckpt, dict) else None
    kwargs = {}
    if size is not None:
        kwargs["size"] = size
    if nb_classes is not None:
        kwargs["nb_classes"] = nb_classes

    if kind == "lcnet":
        print("Detected PP-LCNet-style backbone keys (blocksN.i.dw_conv/pw_conv) -- "
              "this checkpoint needs the PicoDetLCNet subclass, not stock LibrePICODET.")
        if lcnet_script is None:
            raise SystemExit(
                "This checkpoint was trained with a custom PP-LCNet backbone "
                "(e.g. ch_act_picodet_lcnet_backbone.py's PicoDetLCNet). Re-run with:\n"
                "  --lcnet-script /path/to/ch_act_picodet_lcnet_backbone.py\n"
                "(and --lcnet-scale if it wasn't the default 0.75)."
            )
        PicoDetLCNet = import_picodet_lcnet(lcnet_script)
        kwargs["lcnet_scale"] = lcnet_scale
        kwargs.setdefault("size", "s")
        print(f"Using PicoDetLCNet(size='{kwargs.get('size')}', lcnet_scale={lcnet_scale}, "
              f"nc={nb_classes})")
        model = PicoDetLCNet(model_path=weights_path, device=device, **kwargs)

    elif kind == "esnet":
        from libreyolo.models.picodet.model import LibrePICODET
        if size is not None:
            print(f"Using PicoDet size='{size}'"
                  + (f", nc={nb_classes}" if nb_classes is not None else "")
                  + (" (from --size override)" if size_override else " (from checkpoint metadata)"))
        else:
            print("Could not read 'size' from checkpoint metadata; "
                  "falling back to LibrePICODET's default ('s'). "
                  "If loading fails with a shape mismatch, pass --size explicitly.")
        model = LibrePICODET(model_path=weights_path, device=device, **kwargs)

    else:
        raise SystemExit(
            "Could not determine whether this checkpoint uses LibreYOLO's stock ESNet "
            "backbone or a custom PP-LCNet backbone from its key names. "
            "If this is a PicoDetLCNet checkpoint, pass --lcnet-script explicitly; "
            "otherwise this checkpoint may use yet another custom architecture that "
            "these scripts don't recognize."
        )

    model.model.eval()  # defensive no-op; both classes load with eval() already set
    return model


# ---------------------------------------------------------------------------
# 1. Standard COCO-style metrics via LibreYOLO's own validator
# ---------------------------------------------------------------------------

def run_standard_val(model, data, split, imgsz, batch, conf, nms_iou, out_dir):
    metrics = model.val(
        data=data,
        split=split,
        imgsz=imgsz,
        batch=batch,
        conf=conf,
        iou=nms_iou,
        verbose=True,
        save_json=False,
        plots=False,
        save_dir=str(out_dir / "libreyolo_val"),
    )
    return metrics


# ---------------------------------------------------------------------------
# 2. Loss curve from results.csv
# ---------------------------------------------------------------------------

def find_results_csv(weights_path, explicit_path):
    if explicit_path:
        return Path(explicit_path)
    # LibreYOLO writes results.csv in the same run directory as the weights,
    # typically <project>/<name>/weights/last.pt -> <project>/<name>/results.csv
    wp = Path(weights_path).resolve()
    candidate = wp.parent.parent / "results.csv"
    if candidate.is_file():
        return candidate
    candidate2 = wp.parent / "results.csv"
    if candidate2.is_file():
        return candidate2
    return None


def plot_loss_curve(results_csv, out_dir):
    import csv
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if results_csv is None or not Path(results_csv).is_file():
        print(f"[loss curve] results.csv not found (looked for: {results_csv}); skipping.")
        return None

    rows = []
    with open(results_csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({k.strip(): v for k, v in row.items()})

    if not rows:
        print("[loss curve] results.csv is empty; skipping.")
        return None

    cols = rows[0].keys()
    epoch_col = next((c for c in cols if c.lower() in ("epoch", "epochs")), None)
    train_loss_col = next((c for c in cols if c.strip() == "train/loss"), None)
    val_loss_col = next((c for c in cols if "val/loss" in c or c.strip() == "val/loss"), None)

    if epoch_col is None or train_loss_col is None:
        print(f"[loss curve] Could not find epoch/train-loss columns in {results_csv}; "
              f"available columns: {list(cols)}")
        return None

    epochs = [float(r[epoch_col]) for r in rows]
    train_loss = [float(r[train_loss_col]) for r in rows]

    plt.figure(figsize=(7, 5))
    plt.plot(epochs, train_loss, label="train/loss", marker="o", markersize=3)

    if val_loss_col is not None:
        val_loss = []
        for r in rows:
            v = r.get(val_loss_col, "")
            val_loss.append(float(v) if v not in ("", None) else np.nan)
        plt.plot(epochs, val_loss, label="val/loss", marker="o", markersize=3)

    # Also plot loss component breakdown if present (e.g. train/loss/cls, train/loss/box)
    for c in cols:
        if c.startswith("train/loss/"):
            comp = [float(r[c]) if r[c] not in ("", None) else np.nan for r in rows]
            plt.plot(epochs, comp, "--", alpha=0.5, linewidth=1, label=c)

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.legend(fontsize=8)
    plt.grid(alpha=0.3)
    out_path = out_dir / "loss_curve.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[loss curve] saved to {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# 3 & 4. TP/FP/FN, Recall@IoU, Recall@FPPI  (custom matching over raw dets)
# ---------------------------------------------------------------------------

def load_yolo_labels(label_path, img_w, img_h):
    """YOLO txt format: class cx cy w h (normalized). Returns Nx5 array
    [x1, y1, x2, y2, cls] in absolute pixel coords."""
    boxes = []
    if not Path(label_path).is_file():
        return np.zeros((0, 5), dtype=np.float32)
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls, cx, cy, w, h = map(float, parts[:5])
            x1 = (cx - w / 2) * img_w
            y1 = (cy - h / 2) * img_h
            x2 = (cx + w / 2) * img_w
            y2 = (cy + h / 2) * img_h
            boxes.append([x1, y1, x2, y2, cls])
    return np.array(boxes, dtype=np.float32) if boxes else np.zeros((0, 5), dtype=np.float32)


def box_iou(box, boxes):
    """IoU of one box [x1,y1,x2,y2] against an array of boxes (N,4)."""
    if boxes.shape[0] == 0:
        return np.zeros((0,), dtype=np.float32)
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    area_a = (box[2] - box[0]) * (box[3] - box[1])
    area_b = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union = area_a + area_b - inter
    return inter / np.clip(union, 1e-9, None)


def collect_predictions_and_gt(model, data_yaml, split, imgsz, conf, nms_iou, device):
    """Runs the model over every image in the split and returns:
       all_preds: list of (img_idx, x1,y1,x2,y2, conf, cls)
       all_gt:    list of (img_idx, x1,y1,x2,y2, cls)
       n_images:  int
    """
    import yaml
    from libreyolo.data import get_img_files, img2label_paths
    from PIL import Image

    with open(data_yaml) as f:
        data_cfg = yaml.safe_load(f)

    data_dir = Path(data_yaml).parent
    split_path = data_cfg.get(split, data_cfg.get("val"))
    if split_path is None:
        raise ValueError(f"Split '{split}' not found in {data_yaml}")
    split_path = (data_dir / split_path).resolve() if not Path(split_path).is_absolute() else Path(split_path)

    img_files = get_img_files(str(split_path))
    label_files = img2label_paths(img_files)

    all_preds = []  # (img_idx, x1,y1,x2,y2,conf,cls)
    all_gt = []      # (img_idx, x1,y1,x2,y2,cls)
    n_images = len(img_files)

    with torch.no_grad():
        for idx, (img_path, lbl_path) in enumerate(zip(img_files, label_files)):
            img = Image.open(img_path).convert("RGB")
            w, h = img.size

            results = model.predict(
                str(img_path),
                conf=conf,
                iou=nms_iou,
                imgsz=imgsz,
                verbose=False,
                max_det=300,
            )
            result = results[0] if isinstance(results, list) else results
            boxes = result.boxes
            if boxes is not None and len(boxes.xyxy) > 0:
                xyxy = np.asarray(boxes.xyxy.cpu() if hasattr(boxes.xyxy, "cpu") else boxes.xyxy)
                confs = np.asarray(boxes.conf.cpu() if hasattr(boxes.conf, "cpu") else boxes.conf).reshape(-1)
                clses = np.asarray(boxes.cls.cpu() if hasattr(boxes.cls, "cpu") else boxes.cls).reshape(-1)
                for (x1, y1, x2, y2), c, cl in zip(xyxy, confs, clses):
                    all_preds.append((idx, x1, y1, x2, y2, float(c), int(cl)))

            gt_boxes = load_yolo_labels(lbl_path, w, h)
            for x1, y1, x2, y2, cl in gt_boxes:
                all_gt.append((idx, x1, y1, x2, y2, int(cl)))

            if (idx + 1) % 50 == 0 or (idx + 1) == n_images:
                print(f"  [{idx+1}/{n_images}] images processed "
                      f"({len(all_preds)} preds, {len(all_gt)} gt so far)")

    return all_preds, all_gt, n_images


def compute_tp_fp_fn_at_iou(all_preds, all_gt, n_images, iou_thresh, conf_thresh=None):
    """Greedy per-image, per-class matching at a single confidence/IoU
    threshold. Returns dict with tp, fp, fn, precision, recall."""
    preds = [p for p in all_preds if conf_thresh is None or p[5] >= conf_thresh]
    # index gt by image
    gt_by_img = {}
    for (img_idx, x1, y1, x2, y2, cl) in all_gt:
        gt_by_img.setdefault(img_idx, []).append([x1, y1, x2, y2, cl])
    gt_matched = {k: np.zeros(len(v), dtype=bool) for k, v in gt_by_img.items()}

    # sort predictions by confidence descending (standard greedy matching)
    preds_sorted = sorted(preds, key=lambda p: -p[5])

    tp, fp = 0, 0
    for (img_idx, x1, y1, x2, y2, c, cl) in preds_sorted:
        gts = gt_by_img.get(img_idx, [])
        if not gts:
            fp += 1
            continue
        gts_arr = np.array(gts, dtype=np.float32)
        same_cls = gts_arr[:, 4] == cl
        matched_flags = gt_matched[img_idx]
        candidate_mask = same_cls & (~matched_flags)
        if not candidate_mask.any():
            fp += 1
            continue
        ious = box_iou(np.array([x1, y1, x2, y2], dtype=np.float32), gts_arr[:, :4])
        ious = np.where(candidate_mask, ious, -1)
        best_idx = int(np.argmax(ious))
        best_iou = ious[best_idx]
        if best_iou >= iou_thresh:
            tp += 1
            gt_matched[img_idx][best_idx] = True
        else:
            fp += 1

    total_gt = len(all_gt)
    fn = total_gt - tp
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / total_gt if total_gt > 0 else 0.0
    fppi = fp / n_images if n_images > 0 else 0.0
    return dict(tp=tp, fp=fp, fn=fn, total_gt=total_gt, precision=precision,
                recall=recall, fppi=fppi)


def recall_at_fppi(all_preds, all_gt, n_images, iou_thresh, target_fppi, n_thresholds=200):
    """Sweeps confidence threshold, computes FPPI and recall at each point,
    and reports recall at the confidence threshold whose FPPI is closest to
    (and does not exceed, where possible) target_fppi. Also returns the full
    sweep curve for plotting."""
    if not all_preds:
        return dict(recall_at_fppi=0.0, conf_at_fppi=None, curve=[])

    confs = sorted({p[5] for p in all_preds}, reverse=True)
    if len(confs) > n_thresholds:
        idxs = np.linspace(0, len(confs) - 1, n_thresholds).astype(int)
        confs = [confs[i] for i in idxs]

    curve = []
    for ct in confs:
        r = compute_tp_fp_fn_at_iou(all_preds, all_gt, n_images, iou_thresh, conf_thresh=ct)
        curve.append((ct, r["fppi"], r["recall"]))

    # find operating point: highest recall among points with fppi <= target;
    # if none qualify (all fppi too high even at max conf), take the closest.
    eligible = [c for c in curve if c[1] <= target_fppi]
    if eligible:
        best = max(eligible, key=lambda c: c[2])  # max recall among eligible
    else:
        best = min(curve, key=lambda c: abs(c[1] - target_fppi))

    return dict(recall_at_fppi=best[2], conf_at_fppi=best[0], fppi_at_point=best[1], curve=curve)


def plot_fppi_recall_curve(curve, target_fppi, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not curve:
        return None
    fppis = [c[1] for c in curve]
    recalls = [c[2] for c in curve]
    plt.figure(figsize=(7, 5))
    plt.plot(fppis, recalls, marker=".", markersize=3)
    plt.axvline(target_fppi, color="red", linestyle="--", label=f"target FPPI={target_fppi}")
    plt.xlabel("False Positives Per Image (FPPI)")
    plt.ylabel("Recall")
    plt.title("Recall vs FPPI")
    plt.legend()
    plt.grid(alpha=0.3)
    out_path = out_dir / "recall_vs_fppi.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[fppi curve] saved to {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)

    print(f"Loading checkpoint: {args.weights}")
    model = load_model(
        args.weights, device=str(device), size_override=args.size,
        lcnet_script=args.lcnet_script, lcnet_scale=args.lcnet_scale,
    )

    imgsz_hw = parse_imgsz(args.imgsz)
    if imgsz_hw is None:
        native = getattr(model, "input_size", None) or model._get_input_size()
        if isinstance(native, (list, tuple)):
            imgsz_hw = (int(native[0]), int(native[1]))
        else:
            imgsz_hw = (int(native), int(native))
    imgsz = imgsz_hw  # (h, w) tuple, passed through to val()/predict()
    h, w = imgsz_hw
    print(f"Input size: {w}x{h} (WxH) -> internal (h,w)=({h},{w})  |  Device: {device}  |  Split: {args.split}")

    all_metrics = {}

    # 1) Standard metrics (mAP etc.) via LibreYOLO's own validator
    print("\n=== Standard validation metrics ===")
    try:
        std_metrics = run_standard_val(
            model, args.data, args.split, imgsz, args.batch, args.conf, args.nms_iou, out_dir
        )
        all_metrics["standard"] = std_metrics
        for k, v in std_metrics.items():
            print(f"  {k}: {v}")
    except Exception as e:
        print(f"  [standard val] failed: {e}")
        all_metrics["standard"] = {"error": str(e)}

    # 2) Loss curve
    print("\n=== Loss curve ===")
    results_csv = find_results_csv(args.weights, args.results_csv)
    loss_plot_path = plot_loss_curve(results_csv, out_dir)
    all_metrics["loss_curve_plot"] = str(loss_plot_path) if loss_plot_path else None

    # 3) Raw predictions + GT for custom matching (TP/FP/FN, recall@IoU, recall@FPPI)
    print("\n=== Collecting raw predictions + ground truth for matching ===")
    all_preds, all_gt, n_images = collect_predictions_and_gt(
        model, args.data, args.split, imgsz, args.conf, args.nms_iou, device
    )
    print(f"Collected {len(all_preds)} predictions and {len(all_gt)} GT boxes over {n_images} images.")

    print(f"\n=== TP / FP / FN @ IoU={args.match_iou} (conf>={args.conf}) ===")
    tpfpfn = compute_tp_fp_fn_at_iou(all_preds, all_gt, n_images, args.match_iou, conf_thresh=args.conf)
    for k, v in tpfpfn.items():
        print(f"  {k}: {v}")
    all_metrics["tp_fp_fn_at_iou"] = tpfpfn

    print(f"\n=== Recall @ FPPI={args.target_fppi}, IoU={args.match_iou} ===")
    fppi_result = recall_at_fppi(all_preds, all_gt, n_images, args.match_iou, args.target_fppi)
    print(f"  recall_at_fppi   : {fppi_result['recall_at_fppi']:.4f}")
    print(f"  conf_at_operating_point : {fppi_result['conf_at_fppi']}")
    print(f"  actual_fppi_at_point    : {fppi_result.get('fppi_at_point')}")
    all_metrics["recall_at_fppi"] = {
        k: v for k, v in fppi_result.items() if k != "curve"
    }
    plot_fppi_recall_curve(fppi_result["curve"], args.target_fppi, out_dir)

    # Save everything
    metrics_path = out_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2, default=float)
    print(f"\nAll metrics saved to {metrics_path}")
    print(f"Plots saved under {out_dir}")


if __name__ == "__main__":
    main()
