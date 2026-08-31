"""
Run PicoDet ONNX inference on a real image, apply confidence threshold +
NMS, unmap boxes from the 320x480 letterbox canvas back to original image
coordinates, and draw annotated boxes.

Usage:
  python3 annotate_test_image.py weights/last.onnx path/to/image.jpg [--conf 0.3] [--iou 0.45]
"""
import argparse

import cv2
import numpy as np
import onnxruntime as ort

IMAGENET_MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
IMAGENET_STD = np.array([58.395, 57.12, 57.375], dtype=np.float32)

CLASS_NAMES = ["basketball", "volleyball"]  # index 0, 1 -- matches classes.txt order
CLASS_COLORS = [(0, 0, 255), (255, 0, 0)]   # BGR: red for basketball, blue for volleyball


def letterbox(img_bgr, target_h=320, target_w=480, pad_value=114):
    orig_h, orig_w = img_bgr.shape[:2]
    ratio = min(target_h / orig_h, target_w / orig_w)
    new_h, new_w = int(round(orig_h * ratio)), int(round(orig_w * ratio))

    rgb = img_bgr[:, :, ::-1]
    resized = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    canvas = np.full((target_h, target_w, 3), pad_value, dtype=np.uint8)
    canvas[:new_h, :new_w] = resized  # top-left anchor

    return canvas, ratio


def preprocess(canvas_rgb):
    chw = canvas_rgb.astype(np.float32)
    chw = (chw - IMAGENET_MEAN) / IMAGENET_STD
    chw = chw.transpose(2, 0, 1)
    return chw[np.newaxis, ...].astype(np.float32)


def nms(boxes, scores, iou_thresh):
    """Simple NMS. boxes: (N,4) x1y1x2y2, scores: (N,)."""
    idxs = scores.argsort()[::-1]
    keep = []
    while len(idxs) > 0:
        i = idxs[0]
        keep.append(i)
        if len(idxs) == 1:
            break
        rest = idxs[1:]

        xx1 = np.maximum(boxes[i, 0], boxes[rest, 0])
        yy1 = np.maximum(boxes[i, 1], boxes[rest, 1])
        xx2 = np.minimum(boxes[i, 2], boxes[rest, 2])
        yy2 = np.minimum(boxes[i, 3], boxes[rest, 3])

        inter_w = np.maximum(0.0, xx2 - xx1)
        inter_h = np.maximum(0.0, yy2 - yy1)
        inter = inter_w * inter_h

        area_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        area_rest = (boxes[rest, 2] - boxes[rest, 0]) * (boxes[rest, 3] - boxes[rest, 1])
        union = area_i + area_rest - inter
        iou = inter / np.maximum(union, 1e-6)

        idxs = rest[iou <= iou_thresh]
    return keep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("onnx_path")
    ap.add_argument("image_path")
    ap.add_argument("--conf", type=float, default=0.3, help="Confidence threshold")
    ap.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold")
    ap.add_argument("--out", default="/home/megha/libreyolo/tests/tmp/annotated.jpg", help="Where to save the annotated image")
    args = ap.parse_args()

    img_bgr = cv2.imread(args.image_path)
    if img_bgr is None:
        raise SystemExit(f"Could not read image at {args.image_path}")
    orig_h, orig_w = img_bgr.shape[:2]

    canvas, ratio = letterbox(img_bgr, target_h=320, target_w=480)
    input_tensor = preprocess(canvas)

    session = ort.InferenceSession(args.onnx_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: input_tensor})[0]  # (1, N, 4+num_classes)
    preds = outputs[0]  # (N, 4+num_classes)

    boxes = preds[:, :4]
    class_scores = preds[:, 4:]
    num_classes = class_scores.shape[1]

    best_cls = class_scores.argmax(axis=1)
    best_scores = class_scores.max(axis=1)

    print(f"Total raw candidates: {len(preds)}")
    print(f"Score range across all candidates: min={best_scores.min():.4f} max={best_scores.max():.4f}")

    # Filter by confidence
    mask = best_scores >= args.conf
    print(f"Candidates above conf={args.conf}: {mask.sum()}")

    if mask.sum() == 0:
        print(f"No detections above conf={args.conf}. Try a lower --conf (e.g. --conf 0.05) "
              f"to inspect raw candidate quality -- note this checkpoint may not be fully trained yet.")
        # Still save the letterboxed canvas so you can see what the model actually saw
        cv2.imwrite(args.out, cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
        print(f"Saved the letterboxed input canvas (no boxes) to {args.out}")
        return

    boxes = boxes[mask]
    best_cls = best_cls[mask]
    best_scores = best_scores[mask]

    # Per-class NMS
    final_boxes, final_cls, final_scores = [], [], []
    for c in range(num_classes):
        c_mask = best_cls == c
        if not c_mask.any():
            continue
        c_boxes = boxes[c_mask]
        c_scores = best_scores[c_mask]
        keep = nms(c_boxes, c_scores, args.iou)
        final_boxes.append(c_boxes[keep])
        final_cls.append(np.full(len(keep), c))
        final_scores.append(c_scores[keep])

    final_boxes = np.concatenate(final_boxes, axis=0) if final_boxes else np.zeros((0, 4))
    final_cls = np.concatenate(final_cls, axis=0) if final_cls else np.zeros((0,), dtype=int)
    final_scores = np.concatenate(final_scores, axis=0) if final_scores else np.zeros((0,))

    print(f"Final detections after NMS: {len(final_boxes)}")

    # Unletterbox: top-left anchored, single uniform scale, no offset needed
    final_boxes = final_boxes / ratio
    final_boxes[:, [0, 2]] = np.clip(final_boxes[:, [0, 2]], 0, orig_w)
    final_boxes[:, [1, 3]] = np.clip(final_boxes[:, [1, 3]], 0, orig_h)

    # Draw on the ORIGINAL image
    annotated = img_bgr.copy()
    for box, cls, score in zip(final_boxes, final_cls, final_scores):
        x1, y1, x2, y2 = box.astype(int)
        color = CLASS_COLORS[cls % len(CLASS_COLORS)]
        label = f"{CLASS_NAMES[cls]} {score:.2f}"
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(annotated, (x1, y1 - th - 6), (x1 + tw, y1), color, -1)
        cv2.putText(annotated, label, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        print(f"  {label}  box(orig px)=({x1},{y1},{x2},{y2})")

    cv2.imwrite(args.out, annotated)
    print(f"\nSaved annotated image to {args.out}")


if __name__ == "__main__":
    main()
