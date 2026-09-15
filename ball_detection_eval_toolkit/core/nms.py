"""Class-agnostic NMS (single class = ball, so no per-class grouping needed)."""

import numpy as np

from .metrics import box_iou_xyxy


def nms_xyxy(boxes: np.ndarray, scores: np.ndarray, iou_thresh: float = 0.5,
             score_thresh: float = 0.0, max_det: int = 200):
    """
    boxes: (N, 4) xyxy
    scores: (N,)
    Returns (kept_boxes, kept_scores) after score filtering + greedy NMS.
    """
    keep_mask = scores >= score_thresh
    boxes = boxes[keep_mask]
    scores = scores[keep_mask]

    if boxes.shape[0] == 0:
        return boxes, scores

    order = np.argsort(-scores)
    boxes = boxes[order]
    scores = scores[order]

    keep = []
    suppressed = np.zeros(boxes.shape[0], dtype=bool)

    for i in range(boxes.shape[0]):
        if suppressed[i]:
            continue
        keep.append(i)
        if len(keep) >= max_det:
            break
        if i + 1 >= boxes.shape[0]:
            continue
        ious = box_iou_xyxy(boxes[i:i + 1], boxes[i + 1:])[0]
        suppress_idx = np.where(ious > iou_thresh)[0] + (i + 1)
        suppressed[suppress_idx] = True

    keep = np.array(keep, dtype=np.int64)
    return boxes[keep], scores[keep]
