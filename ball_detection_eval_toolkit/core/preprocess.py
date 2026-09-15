"""
Preprocessing shared by all three backends (ONNX / NanoDet-Plus .pt / PicoDet).

Client requirement (from slides):
    - Input: 320x480 (or 480x320) RGB.
    - "Not all data have the same size as input. Please do not change aspect
       ratio when resizing. You can pad images if not in right aspect ratio."
    => letterbox resize (aspect-ratio-preserving resize + pad), NOT a naive stretch.

All coordinate transforms are tracked via a `LetterboxTransform` so predicted
boxes can be mapped back to ORIGINAL image pixel coordinates before scoring,
since ground truth boxes are stored in original-image pixel space.
"""

from dataclasses import dataclass
from typing import Tuple

import cv2
import numpy as np


@dataclass
class LetterboxTransform:
    scale: float
    pad_x: float   # left padding, in resized-image pixels
    pad_y: float   # top padding, in resized-image pixels
    orig_w: int
    orig_h: int
    dst_w: int
    dst_h: int

    def to_model_xyxy(self, boxes_xyxy_orig: np.ndarray) -> np.ndarray:
        """Map ORIGINAL-image xyxy boxes into model-input pixel space."""
        b = boxes_xyxy_orig.copy()
        b[:, [0, 2]] = b[:, [0, 2]] * self.scale + self.pad_x
        b[:, [1, 3]] = b[:, [1, 3]] * self.scale + self.pad_y
        return b

    def to_orig_xyxy(self, boxes_xyxy_model: np.ndarray) -> np.ndarray:
        """Map model-input-space xyxy boxes back to ORIGINAL image pixels."""
        if boxes_xyxy_model.shape[0] == 0:
            return boxes_xyxy_model.copy()
        b = boxes_xyxy_model.copy().astype(np.float32)
        b[:, [0, 2]] = (b[:, [0, 2]] - self.pad_x) / self.scale
        b[:, [1, 3]] = (b[:, [1, 3]] - self.pad_y) / self.scale
        b[:, [0, 2]] = np.clip(b[:, [0, 2]], 0, self.orig_w - 1)
        b[:, [1, 3]] = np.clip(b[:, [1, 3]], 0, self.orig_h - 1)
        return b


def letterbox_resize(image: np.ndarray, dst_w: int, dst_h: int,
                      pad_value: int = 114) -> Tuple[np.ndarray, LetterboxTransform]:
    """
    Aspect-ratio-preserving resize + center pad.

    image: HWC RGB uint8
    dst_w, dst_h: target model input size (width, height)

    Returns padded image of shape (dst_h, dst_w, 3) and the transform needed
    to map coordinates between original <-> model space.
    """
    orig_h, orig_w = image.shape[:2]
    scale = min(dst_w / orig_w, dst_h / orig_h)
    new_w, new_h = int(round(orig_w * scale)), int(round(orig_h * scale))

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    pad_x = (dst_w - new_w) / 2.0
    pad_y = (dst_h - new_h) / 2.0
    left, top = int(round(pad_x - 0.1)), int(round(pad_y - 0.1))
    right, bottom = dst_w - new_w - left, dst_h - new_h - top

    padded = cv2.copyMakeBorder(resized, top, bottom, left, right,
                                 cv2.BORDER_CONSTANT, value=(pad_value,) * 3)

    transform = LetterboxTransform(
        scale=scale, pad_x=left, pad_y=top,
        orig_w=orig_w, orig_h=orig_h, dst_w=dst_w, dst_h=dst_h,
    )
    return padded, transform


def normalize(image_rgb_uint8: np.ndarray, mean, std, to_chw: bool = True) -> np.ndarray:
    """
    image_rgb_uint8: HWC uint8
    mean, std: length-3 sequences in RGB order, in the SAME scale as the
               desired output (e.g. mean=[0,0,0], std=[255,255,255] for
               plain 0-1 scaling, or ImageNet stats for mean/std normalization).
    """
    img = image_rgb_uint8.astype(np.float32)
    mean = np.array(mean, dtype=np.float32).reshape(1, 1, 3)
    std = np.array(std, dtype=np.float32).reshape(1, 1, 3)
    img = (img - mean) / std
    if to_chw:
        img = img.transpose(2, 0, 1)  # HWC -> CHW
    return np.ascontiguousarray(img)


def xywh_to_xyxy(boxes_xywh: np.ndarray) -> np.ndarray:
    if boxes_xywh.shape[0] == 0:
        return boxes_xywh.reshape(0, 4).astype(np.float32)
    x, y, w, h = boxes_xywh[:, 0], boxes_xywh[:, 1], boxes_xywh[:, 2], boxes_xywh[:, 3]
    return np.stack([x, y, x + w, y + h], axis=1).astype(np.float32)


def xyxy_to_xywh(boxes_xyxy: np.ndarray) -> np.ndarray:
    if boxes_xyxy.shape[0] == 0:
        return boxes_xyxy.reshape(0, 4).astype(np.float32)
    x1, y1, x2, y2 = boxes_xyxy[:, 0], boxes_xyxy[:, 1], boxes_xyxy[:, 2], boxes_xyxy[:, 3]
    return np.stack([x1, y1, x2 - x1, y2 - y1], axis=1).astype(np.float32)


def cxcywh_to_xyxy(boxes_cxcywh: np.ndarray) -> np.ndarray:
    if boxes_cxcywh.shape[0] == 0:
        return boxes_cxcywh.reshape(0, 4).astype(np.float32)
    cx, cy, w, h = (boxes_cxcywh[:, 0], boxes_cxcywh[:, 1],
                     boxes_cxcywh[:, 2], boxes_cxcywh[:, 3])
    return np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1).astype(np.float32)
