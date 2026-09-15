"""
Evaluation metrics matching the client's stated spec:

    Evaluation Metrics: Recall rate at FPPI=0.05 with IoU > 0.4, with given test data
    Accuracy expectation: at least > 60%

Definitions used here (standard single-class detector evaluation):
    - A prediction is a True Positive (TP) if its IoU with an UNMATCHED ground
      truth box exceeds the IoU threshold (0.4). Each GT box can be matched
      at most once (highest-confidence prediction wins ties).
    - A prediction that matches no GT box is a False Positive (FP).
    - A GT box with no matching prediction is a False Negative (FN).
    - FPPI (False Positives Per Image) = total FP / total number of images,
      computed at a given confidence threshold.
    - Recall @ FPPI=X: sweep confidence threshold from high to low, find the
      lowest threshold at which FPPI <= X, report recall at that point.
      (Equivalently: the recall value on the FPPI-vs-recall curve where the
      curve first reaches FPPI = X, interpolating between adjacent points.)

Also reports:
    - Sports-wise recall (same metric, computed per sport subset)
    - Full FPPI/recall curve data (for plotting / report appendix)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


def box_iou_xyxy(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    """
    boxes1: (N, 4) xyxy
    boxes2: (M, 4) xyxy
    returns: (N, M) IoU matrix
    """
    if boxes1.shape[0] == 0 or boxes2.shape[0] == 0:
        return np.zeros((boxes1.shape[0], boxes2.shape[0]), dtype=np.float32)

    area1 = (boxes1[:, 2] - boxes1[:, 0]).clip(0) * (boxes1[:, 3] - boxes1[:, 1]).clip(0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clip(0) * (boxes2[:, 3] - boxes2[:, 1]).clip(0)

    lt = np.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = np.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = (rb - lt).clip(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]

    union = area1[:, None] + area2[None, :] - inter
    iou = np.where(union > 0, inter / union, 0.0)
    return iou.astype(np.float32)


@dataclass
class PerImagePrediction:
    image_id: str
    sport: str
    pred_boxes_xyxy: np.ndarray   # (P, 4), original-image pixel space
    pred_scores: np.ndarray       # (P,)
    gt_boxes_xyxy: np.ndarray     # (G, 4), original-image pixel space


@dataclass
class MatchResult:
    """Flattened match records across the whole dataset, used to build curves."""
    scores: np.ndarray            # (K,) confidence of each prediction
    is_tp: np.ndarray             # (K,) bool, True if that prediction is a TP
    sport: List[str]              # (K,) sport tag per prediction
    n_images: int
    n_gt_total: int
    n_gt_by_sport: Dict[str, int]
    n_images_by_sport: Dict[str, int]


def match_predictions(records: List[PerImagePrediction], iou_thresh: float = 0.4) -> MatchResult:
    all_scores = []
    all_is_tp = []
    all_sport = []
    n_gt_total = 0
    n_gt_by_sport: Dict[str, int] = {}
    n_images_by_sport: Dict[str, int] = {}

    for rec in records:
        n_gt_total += rec.gt_boxes_xyxy.shape[0]
        n_gt_by_sport[rec.sport] = n_gt_by_sport.get(rec.sport, 0) + rec.gt_boxes_xyxy.shape[0]
        n_images_by_sport[rec.sport] = n_images_by_sport.get(rec.sport, 0) + 1

        if rec.pred_boxes_xyxy.shape[0] == 0:
            continue

        order = np.argsort(-rec.pred_scores)
        pred_boxes = rec.pred_boxes_xyxy[order]
        pred_scores = rec.pred_scores[order]

        gt_matched = np.zeros(rec.gt_boxes_xyxy.shape[0], dtype=bool)

        if rec.gt_boxes_xyxy.shape[0] > 0:
            ious = box_iou_xyxy(pred_boxes, rec.gt_boxes_xyxy)  # (P, G)
        else:
            ious = np.zeros((pred_boxes.shape[0], 0), dtype=np.float32)

        for i in range(pred_boxes.shape[0]):
            is_tp = False
            if ious.shape[1] > 0:
                cand = np.where(~gt_matched)[0]
                if cand.size > 0:
                    cand_ious = ious[i, cand]
                    best_local = np.argmax(cand_ious)
                    best_iou = cand_ious[best_local]
                    if best_iou > iou_thresh:
                        gt_matched[cand[best_local]] = True
                        is_tp = True
            all_scores.append(pred_scores[i])
            all_is_tp.append(is_tp)
            all_sport.append(rec.sport)

    return MatchResult(
        scores=np.array(all_scores, dtype=np.float32) if all_scores else np.zeros(0, dtype=np.float32),
        is_tp=np.array(all_is_tp, dtype=bool) if all_is_tp else np.zeros(0, dtype=bool),
        sport=all_sport,
        n_images=len(records),
        n_gt_total=n_gt_total,
        n_gt_by_sport=n_gt_by_sport,
        n_images_by_sport=n_images_by_sport,
    )


def recall_at_fppi(match: MatchResult, target_fppi: float = 0.05,
                    n_gt_override: Optional[int] = None,
                    n_images_override: Optional[int] = None,
                    sport: Optional[str] = None) -> dict:
    """
    Computes Recall @ FPPI = target_fppi by sweeping the confidence threshold.

    If `sport` is given, restricts to that sport's predictions/GT counts
    (uses n_gt_by_sport / n_images_by_sport from match unless overridden).

    Returns a dict with the operating point plus the full curve for plotting.
    """
    if sport is not None:
        mask = np.array([s == sport for s in match.sport], dtype=bool)
        scores = match.scores[mask]
        is_tp = match.is_tp[mask]
        n_gt = match.n_gt_by_sport.get(sport, 0)
        n_images = match.n_images_by_sport.get(sport, 0)
    else:
        scores = match.scores
        is_tp = match.is_tp
        n_gt = n_gt_override if n_gt_override is not None else match.n_gt_total
        n_images = n_images_override if n_images_override is not None else match.n_images

    if n_gt == 0 or n_images == 0 or scores.shape[0] == 0:
        return {
            "recall_at_target_fppi": 0.0,
            "operating_threshold": None,
            "target_fppi": target_fppi,
            "n_gt": n_gt,
            "n_images": n_images,
            "curve": {"thresholds": [], "recall": [], "fppi": []},
            "note": "No predictions or no ground truth in this subset.",
        }

    order = np.argsort(-scores)
    scores_sorted = scores[order]
    is_tp_sorted = is_tp[order]

    tp_cum = np.cumsum(is_tp_sorted)
    fp_cum = np.cumsum(~is_tp_sorted)

    recall_curve = tp_cum / n_gt
    fppi_curve = fp_cum / n_images

    # Find operating point: lowest-confidence-threshold position where
    # fppi_curve <= target_fppi (curve is monotonically non-decreasing in FPPI
    # as threshold decreases, i.e. as we include more predictions).
    valid_idx = np.where(fppi_curve <= target_fppi)[0]

    if valid_idx.size == 0:
        # Even the single highest-confidence prediction already exceeds target FPPI
        recall_at_target = 0.0
        threshold = float(scores_sorted[0]) if scores_sorted.size > 0 else None
    else:
        best_idx = valid_idx[-1]  # last index still satisfying fppi <= target
        recall_at_target = float(recall_curve[best_idx])
        threshold = float(scores_sorted[best_idx])

    return {
        "recall_at_target_fppi": recall_at_target,
        "operating_threshold": threshold,
        "target_fppi": target_fppi,
        "n_gt": int(n_gt),
        "n_images": int(n_images),
        "curve": {
            "thresholds": scores_sorted.tolist(),
            "recall": recall_curve.tolist(),
            "fppi": fppi_curve.tolist(),
        },
    }


def evaluate(records: List[PerImagePrediction], iou_thresh: float = 0.4,
             target_fppi: float = 0.05) -> dict:
    """
    Full evaluation entry point. Returns overall + per-sport Recall@FPPI,
    matching client's "Numbers Expected in Report" items 1 and 2.
    """
    match = match_predictions(records, iou_thresh=iou_thresh)
    overall = recall_at_fppi(match, target_fppi=target_fppi)

    sports_present = sorted(set(match.sport)) if match.sport else []
    per_sport = {}
    for sport in sports_present:
        per_sport[sport] = recall_at_fppi(match, target_fppi=target_fppi, sport=sport)

    return {
        "iou_threshold": iou_thresh,
        "target_fppi": target_fppi,
        "overall": overall,
        "per_sport": per_sport,
        "n_images_total": match.n_images,
        "n_gt_total": match.n_gt_total,
    }


def collect_fp_fn_samples(records: List[PerImagePrediction], threshold: float,
                           iou_thresh: float = 0.4, max_samples: int = 20) -> dict:
    """
    Re-runs matching AT A FIXED confidence threshold (typically the operating
    threshold from recall_at_fppi) to collect concrete false-positive and
    false-negative examples for the report (item #5).
    """
    fps = []
    fns = []

    for rec in records:
        keep = rec.pred_scores >= threshold
        pred_boxes = rec.pred_boxes_xyxy[keep]
        pred_scores = rec.pred_scores[keep]

        gt_matched = np.zeros(rec.gt_boxes_xyxy.shape[0], dtype=bool)

        if pred_boxes.shape[0] > 0:
            order = np.argsort(-pred_scores)
            pred_boxes = pred_boxes[order]
            pred_scores = pred_scores[order]

        if rec.gt_boxes_xyxy.shape[0] > 0 and pred_boxes.shape[0] > 0:
            ious = box_iou_xyxy(pred_boxes, rec.gt_boxes_xyxy)
        else:
            ious = np.zeros((pred_boxes.shape[0], rec.gt_boxes_xyxy.shape[0]), dtype=np.float32)

        for i in range(pred_boxes.shape[0]):
            is_tp = False
            if ious.shape[1] > 0:
                cand = np.where(~gt_matched)[0]
                if cand.size > 0:
                    cand_ious = ious[i, cand]
                    best_local = np.argmax(cand_ious)
                    best_iou = cand_ious[best_local]
                    if best_iou > iou_thresh:
                        gt_matched[cand[best_local]] = True
                        is_tp = True
            if not is_tp and len(fps) < max_samples:
                fps.append({
                    "image_id": rec.image_id,
                    "sport": rec.sport,
                    "box_xyxy": pred_boxes[i].tolist(),
                    "score": float(pred_scores[i]),
                })

        for g in range(rec.gt_boxes_xyxy.shape[0]):
            if not gt_matched[g] and len(fns) < max_samples:
                fns.append({
                    "image_id": rec.image_id,
                    "sport": rec.sport,
                    "gt_box_xyxy": rec.gt_boxes_xyxy[g].tolist(),
                })

        if len(fps) >= max_samples and len(fns) >= max_samples:
            break

    return {"false_positives": fps, "false_negatives": fns, "threshold_used": threshold}
