"""Evaluate an exported PicoDet ONNX model -- same TP/FP/FN, multi-IoU sweep,
recall@FPPI, and annotated FP/FN images as picodet_eval.py, but running the ONNX
Runtime graph instead of the PyTorch model. Reuses picodet_eval.py's metric and
visualization code directly (imported, not duplicated) so both scripts score things
identically and any fix to one applies to both.

The ONNX graph outputs raw dense predictions -- (1, N, 4+nc), canvas-pixel boxes,
already-sigmoid per-class scores, NO NMS applied (see export_onnx.py). This script
does the decode -> per-class NMS -> un-letterbox steps in plain numpy, so it exercises
nothing from the PyTorch/libreyolo postprocessing path -- a genuine check of the
exported artifact, not a re-test of the .pt model.

Usage:
    python picodet_eval_onnx.py --data dataset.yaml --onnx picodet_esnet_320x480.onnx \\
        --imgsz 320 480 --iou 0.5 --conf 0.25 --fppi-iou 0.4 --fppi-target 0.05 --vis 12
"""
import argparse
import os

import numpy as np
import onnxruntime as ort

from picodet_eval import (
    iou_matrix, iou_sweep, load_ground_truth, recall_at_fppi,
    report_at_threshold, save_fp_fn_samples,
)
from picodet_letterbox import letterbox_np

# Preprocessing conventions for different exported models. PicoDet/LibreYOLO converts
# BGR->RGB then subtracts an RGB-ordered ImageNet mean. NanoDet-Plus (and many
# mmdetection-lineage configs) do NOT convert to RGB -- they normalize directly in the
# BGR order cv2 loads images in, with a BGR-ordered mean/std to match. Same three
# ImageNet constants, opposite order -- confirmed against a real NanoDet-Plus training
# config (ball_exp2_ch48_320x480.yml): mean=[103.53,116.28,123.675], std=[57.375,57.12,
# 58.395], BGR order, no channel swap. Getting this wrong doesn't crash -- it silently
# swaps red and blue on every image and gives plausible-looking wrong numbers.
PREPROC_PRESETS = {
    "picodet": {"bgr": False, "mean": (123.675, 116.28, 103.53), "std": (58.395, 57.12, 57.375)},
    "nanodetplus": {"bgr": True, "mean": (103.53, 116.28, 123.675), "std": (57.375, 57.12, 58.395)},
}


def numpy_nms(boxes: np.ndarray, scores: np.ndarray, iou_thres: float) -> np.ndarray:
    """Standard greedy NMS, single class. Returns indices to keep, highest score first."""
    order = scores.argsort()[::-1]
    keep = []
    while len(order) > 0:
        i = order[0]
        keep.append(i)
        if len(order) == 1:
            break
        ious = iou_matrix(boxes[i][None, :], boxes[order[1:]])[0]
        order = order[1:][ious < iou_thres]
    return np.array(keep, dtype=int)


def decode_onnx_output(raw: np.ndarray, conf_thres: float, nms_iou: float, max_det: int) -> np.ndarray:
    """raw: (N, 4+nc) for ONE image (canvas pixel boxes, per-class sigmoid scores).
    Returns [x1,y1,x2,y2,score,cls], one row per kept detection, all classes combined."""
    boxes = raw[:, :4]
    class_scores = raw[:, 4:]
    nc = class_scores.shape[1]

    all_dets = []
    for c in range(nc):
        scores_c = class_scores[:, c]
        mask = scores_c >= conf_thres
        if not mask.any():
            continue
        b, s = boxes[mask], scores_c[mask]
        keep = numpy_nms(b, s, nms_iou)
        for i in keep:
            all_dets.append([b[i, 0], b[i, 1], b[i, 2], b[i, 3], s[i], c])

    if not all_dets:
        return np.zeros((0, 6), dtype=np.float32)
    dets = np.array(all_dets, dtype=np.float32)
    if len(dets) > max_det:
        dets = dets[np.argsort(-dets[:, 4])[:max_det]]
    return dets


def run_onnx_inference(onnx_path: str, images: dict, imgsz, conf_floor: float,
                       nms_iou: float, max_det: int, preproc: str = "picodet"):
    """Same contract as picodet_eval.run_inference: returns image_id -> [x1,y1,x2,y2,score,cls]
    in ORIGINAL image pixel coordinates. ``preproc`` selects a PREPROC_PRESETS entry --
    use "nanodetplus" for a NanoDet-Plus export, "picodet" (default) for PicoDet."""
    from libreyolo.utils.image_loader import ImageLoader

    cfg = PREPROC_PRESETS[preproc]
    mean = np.array(cfg["mean"], dtype=np.float32)
    std = np.array(cfg["std"], dtype=np.float32)

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    target_h, target_w = imgsz

    preds = {}
    for img_id, path in images.items():
        img = ImageLoader.load(path, color_format="auto")
        orig_w, orig_h = img.size
        img_arr = np.array(img)  # RGB, from PIL
        if cfg["bgr"]:
            img_arr = img_arr[:, :, ::-1]  # RGB -> BGR, matching NanoDet-Plus's own convention
        canvas, ratio, _, _ = letterbox_np(img_arr, imgsz)
        arr = canvas.astype(np.float32)
        arr = (arr - mean) / std
        chw = arr.transpose(2, 0, 1)[None, ...].astype(np.float32)

        raw = sess.run(None, {input_name: chw})[0][0]  # drop batch dim -> (N, 4+nc)
        dets = decode_onnx_output(raw, conf_floor, nms_iou, max_det)
        if len(dets):
            dets[:, :4] /= ratio  # top-left pad -> no offset, same convention as everywhere else
            dets[:, [0, 2]] = dets[:, [0, 2]].clip(0, orig_w)
            dets[:, [1, 3]] = dets[:, [1, 3]].clip(0, orig_h)
        preds[img_id] = dets
    return preds


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--split", default="val")
    p.add_argument("--onnx", required=True)
    p.add_argument("--imgsz", type=int, nargs=2, default=[320, 480], metavar=("H", "W"),
                   help="MUST match the size export_onnx.py baked into this graph")
    p.add_argument("--preproc", default="picodet", choices=sorted(PREPROC_PRESETS),
                   help="preprocessing convention: 'picodet' (RGB, LibreYOLO's ImageNet "
                        "mean/std) or 'nanodetplus' (raw BGR, NanoDet-Plus's mean/std). "
                        "Using the wrong one silently swaps red/blue -- see module docstring.")
    p.add_argument("--iou", type=float, default=0.5)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou-sweep", type=float, nargs="+", default=[0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.9])
    p.add_argument("--fppi-iou", type=float, default=0.4)
    p.add_argument("--fppi-target", type=float, default=0.05)
    p.add_argument("--conf-floor", type=float, default=0.001)
    p.add_argument("--nms-iou", type=float, default=0.45)
    p.add_argument("--max-det", type=int, default=300)
    p.add_argument("--vis", type=int, default=10)
    p.add_argument("--out", default="eval_out_onnx")
    args = p.parse_args()

    images, gts, class_names = load_ground_truth(args.data, args.split)
    print(f"{args.split}: {len(images)} images, {sum(len(g) for g in gts.values())} GT boxes, "
          f"classes={class_names}")

    preds = run_onnx_inference(args.onnx, images, tuple(args.imgsz), args.conf_floor,
                               args.nms_iou, args.max_det, preproc=args.preproc)

    report_at_threshold(preds, gts, args.iou, args.conf, class_names)
    iou_sweep(preds, gts, args.iou_sweep, args.conf)
    recall_at_fppi(preds, gts, args.fppi_iou, args.fppi_target)

    if args.vis > 0:
        os.makedirs(args.out, exist_ok=True)
        save_fp_fn_samples(images, preds, gts, args.iou, args.conf, class_names,
                          os.path.join(args.out, "fp_fn_samples"), args.vis)


if __name__ == "__main__":
    main()
