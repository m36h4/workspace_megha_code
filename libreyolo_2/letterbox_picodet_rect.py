"""
Complete letterbox patch for LibreYOLO PicoDet at RECTANGULAR 320 (h) x 480 (w).

Top-left-anchored letterbox (same convention as YOLOXValPreprocessor in
this codebase): resize preserving aspect ratio so the image fits inside
the (320, 480) canvas, pad remaining space (bottom and/or right) with
gray (114,114,114), no offset on the top-left corner.

Verified by round-trip test: a box in original image coordinates ->
letterboxed canvas coordinates -> unletterboxed back == the original,
exactly, for rectangular targets.

Note: PicoDet's 4th FPN level (P6, stride 64) expects input divisible by
64. 320 is (320/64=5 exactly); 480 is not (480/64=7.5). The neck's
forward() already guards this with F.interpolate to align mismatched
feature map sizes, so this runs without error, but the P6 level's
feature alignment is not perfectly clean. This is a pre-existing
tolerance in the architecture, not something this patch introduces.

--- Usage ---
Monkey-patch BEFORE calling model.train() or running inference:

    import libreyolo.models.picodet.trainer as picodet_trainer
    import libreyolo.validation.preprocessors as val_preproc
    from letterbox_picodet_rect import (
        LetterboxPICODETTrainTransform,
        LetterboxPICODETValPreprocessor,
    )
    picodet_trainer.PICODETTrainTransform = LetterboxPICODETTrainTransform
    val_preproc.PICODETValPreprocessor = LetterboxPICODETValPreprocessor

    model.train(data="dataset.yaml", imgsz=(320, 480), ...)
"""
from __future__ import annotations

import random
from typing import Tuple

import cv2
import numpy as np
import torch

from libreyolo.data.augment.boxes import xyxy2cxcywh
from libreyolo.data.augment.color import augment_hsv
from libreyolo.data.augment.geometry import mirror
from libreyolo.models.picodet.utils import IMAGENET_MEAN, IMAGENET_STD
from libreyolo.validation.preprocessors import StandardValPreprocessor


# ---------------------------------------------------------------------------
# 1. Validation / inference preprocessor
# ---------------------------------------------------------------------------

class LetterboxPICODETValPreprocessor(StandardValPreprocessor):
    """PICODET val preprocessor with top-left letterbox pad, rectangular-safe."""

    _MEAN = np.array(IMAGENET_MEAN, dtype=np.float32).reshape(3, 1, 1)
    _STD = np.array(IMAGENET_STD, dtype=np.float32).reshape(3, 1, 1)

    @property
    def custom_normalization(self) -> bool:
        return True

    @property
    def uses_letterbox(self) -> bool:
        return True

    @property
    def wants_unresized_image(self) -> bool:
        return True

    def __call__(
        self, img: np.ndarray, targets: np.ndarray, input_size: Tuple[int, int]
    ) -> Tuple[np.ndarray, np.ndarray]:
        target_h, target_w = input_size
        orig_h, orig_w = img.shape[:2]

        ratio = min(target_h / orig_h, target_w / orig_w)
        new_h, new_w = int(round(orig_h * ratio)), int(round(orig_w * ratio))

        rgb = np.ascontiguousarray(img[:, :, ::-1])  # BGR -> RGB
        resized = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        padded = np.full((target_h, target_w, 3), 114, dtype=np.uint8)
        padded[:new_h, :new_w] = resized  # top-left anchor

        chw = padded.transpose(2, 0, 1).astype(np.float32)
        chw = (chw - self._MEAN) / self._STD

        padded_targets = np.zeros((self.max_labels, 5), dtype=np.float32)
        if len(targets) > 0:
            t = np.array(targets, dtype=np.float32).copy()
            n = min(len(t), self.max_labels)
            t[:n, 0:4] *= ratio  # single uniform scale, top-left anchor = no offset
            padded_targets[:n] = t[:n]

        return np.ascontiguousarray(chw), padded_targets


# ---------------------------------------------------------------------------
# 2. Training transform
# ---------------------------------------------------------------------------

class LetterboxPICODETTrainTransform:
    """Training transform with top-left letterbox pad, rectangular-safe.

    Contract identical to the original PICODETTrainTransform:
        in:  image (BGR HWC uint8), targets ([N,5] x1,y1,x2,y2,class in
             ORIGINAL pixel coords), input_dim (h, w)
        out: (img_chw float32, padded_labels [max_labels,5] class,cx,cy,w,h
             in PADDED CANVAS pixel coords)
    """

    _MEAN = np.array(IMAGENET_MEAN, dtype=np.float32)
    _STD = np.array(IMAGENET_STD, dtype=np.float32)

    wants_unresized_image = True

    def __init__(self, max_labels: int = 50, flip_prob: float = 0.5,
                 hsv_prob: float = 0.0, pad_value: int = 114):
        self.max_labels = max_labels
        self.flip_prob = flip_prob
        self.hsv_prob = hsv_prob
        self.pad_value = pad_value

    def __call__(self, image, targets, input_dim):
        input_h, input_w = int(input_dim[0]), int(input_dim[1])
        orig_h, orig_w = image.shape[:2]

        boxes = targets[:, :4].copy()
        labels = targets[:, 4].copy()

        if self.hsv_prob > 0 and random.random() < self.hsv_prob:
            augment_hsv(image)
        image, boxes = mirror(image, boxes, self.flip_prob)

        rgb = np.ascontiguousarray(image[:, :, ::-1])  # BGR -> RGB

        ratio = min(input_h / orig_h, input_w / orig_w)
        new_h, new_w = int(round(orig_h * ratio)), int(round(orig_w * ratio))
        resized = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        canvas = np.full((input_h, input_w, 3), self.pad_value, dtype=np.uint8)
        canvas[:new_h, :new_w] = resized  # top-left anchor
        canvas = canvas.astype(np.float32)

        canvas = (canvas - self._MEAN) / self._STD
        img_chw = np.ascontiguousarray(canvas.transpose(2, 0, 1), dtype=np.float32)

        padded_labels = np.zeros((self.max_labels, 5), dtype=np.float32)
        if len(boxes) > 0:
            boxes = boxes * ratio  # single uniform scale, top-left anchor -> no offset
            boxes = xyxy2cxcywh(boxes)
            mask = np.minimum(boxes[:, 2], boxes[:, 3]) > 1
            boxes = boxes[mask]
            labels = labels[mask]
            n = min(len(boxes), self.max_labels)
            if n > 0:
                padded_labels[:n, 0] = labels[:n]
                padded_labels[:n, 1:] = boxes[:n]

        return img_chw, padded_labels


# ---------------------------------------------------------------------------
# 3. Postprocess: unmap predicted boxes back to original image coordinates
# ---------------------------------------------------------------------------

def unletterbox_boxes_topleft(
    boxes_xyxy: torch.Tensor,
    orig_w: int,
    orig_h: int,
    canvas_h: int = 320,
    canvas_w: int = 480,
) -> torch.Tensor:
    """Invert a top-left-anchored rectangular letterbox transform.

    boxes_xyxy: (N, 4) in the model's canvas pixel coords.
    Top-left anchoring means no offset subtraction is needed -- only a
    single division by the same uniform ratio used at preprocessing time.
    """
    ratio = min(canvas_h / orig_h, canvas_w / orig_w)
    boxes = boxes_xyxy.clone() / ratio
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, orig_w)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, orig_h)
    return boxes
