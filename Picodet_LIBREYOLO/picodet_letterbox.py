"""PicoDet augmentation matching PaddleDetection's actual training recipe, with letterbox
resizing swapped in for Paddle's stretch resize.

Verified against PaddleDetection's real LCNet PicoDet config (picodet_s_320_lcnet_pedestrian.yml,
release/2.5) and ppdet/data/transform/operators.py:

    TrainReader:
      sample_transforms: [Decode, RandomCrop, RandomFlip(prob=0.5), RandomDistort]
      batch_transforms: [BatchRandomResize(sizes=[256,288,320,352,384], keep_ratio=False), Normalize]

IMPORTANT -- this deliberately does NOT reproduce Paddle exactly, on your instruction:
  1. Paddle's own BatchRandomResize uses keep_ratio=False (stretch), same as LibreYOLO's
     existing default. Paddle does not letterbox PicoDet anywhere. You asked for letterbox
     specifically, so RandomCrop/RandomDistort are ported faithfully (same formulas, same
     default parameters as Paddle) but the final resize step is letterbox instead of stretch.
  2. Paddle's multi-scale resize is applied per BATCH (the whole batch shares one size that
     step) via a batch-level transform. LibreYOLO's PicoDet transform runs per SAMPLE (in
     each worker's __getitem__), with no batch-level hook available. Reproducing true
     per-batch resize would need a custom collate_fn and was out of scope here. Instead,
     each sample independently picks a smaller canvas from a scaled-down version of Paddle's
     size list, letterboxes into it, then pads up to the fixed training canvas -- so every
     tensor in a batch is still the same shape (required for stacking), but different images
     are trained at different effective resolutions. This gives real scale diversity, just
     distributed across samples instead of across batches.

RandomCrop and RandomDistort use Paddle's exact default parameters and formulas (aspect_ratio,
thresholds, scaling, the YIQ-space hue rotation, etc.) - see the operator source for both:
https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.6/ppdet/data/transform/operators.py

Usage:
    import picodet_letterbox  # patches PicoDet before you build the model
    from libreyolo.models.picodet.model import LibrePICODET
    model = LibrePICODET(size="s")
    model.train(data="dataset.yaml", imgsz=(320, 448), epochs=300, ...)

Toggle pieces off if you want to isolate their effect:
    import picodet_letterbox
    picodet_letterbox.RANDOM_CROP = False     # keep letterbox + flip only
    picodet_letterbox.RANDOM_DISTORT = False
    picodet_letterbox.MULTISCALE = False      # always letterbox straight to the full canvas
(set before building the model - the transform reads these at construction time)
"""
from __future__ import annotations

import random
from typing import Tuple

import cv2
import numpy as np
import torch

from libreyolo.data.augment.geometry import mirror
from libreyolo.models.picodet import trainer as _picodet_trainer
from libreyolo.models.picodet import utils as _picodet_utils
from libreyolo.models.picodet.model import LibrePICODET
from libreyolo.models.picodet.utils import IMAGENET_MEAN, IMAGENET_STD
from libreyolo.postprocess.picodet import postprocess as _picodet_decode
from libreyolo.utils.image_loader import ImageLoader
from libreyolo.validation.preprocessors import BaseValPreprocessor

PAD_VALUE = 114  # mid-gray, same convention as YOLO9's letterbox in this codebase

# Toggles - flip to False to isolate one piece's effect. Read at transform construction time.
RANDOM_CROP = True
RANDOM_DISTORT = True
MULTISCALE = True
MULTISCALE_FRACTIONS = (0.7, 0.8, 0.9, 1.0, 1.0)  # weighted toward the full deploy canvas


def _target_hw(imgsz) -> Tuple[int, int]:
    if isinstance(imgsz, (list, tuple)):
        return int(imgsz[0]), int(imgsz[1])
    return int(imgsz), int(imgsz)


def letterbox_np(img: np.ndarray, target_hw, pad_value: int = PAD_VALUE):
    """Top-left letterbox: resize by a single ratio, pad the rest.

    Returns (canvas, ratio, new_h, new_w). ``img`` is HWC, any channel order.
    """
    target_h, target_w = _target_hw(target_hw)
    orig_h, orig_w = img.shape[:2]
    ratio = min(target_h / orig_h, target_w / orig_w)
    new_h, new_w = int(round(orig_h * ratio)), int(round(orig_w * ratio))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((target_h, target_w, img.shape[2]), pad_value, dtype=img.dtype)
    canvas[:new_h, :new_w] = resized
    return canvas, ratio, new_h, new_w


# ---------------------------------------------------------------------------
# RandomCrop - ported from ppdet's RandomCrop, same defaults, same algorithm.
# Operates on (image, boxes_xyxy, labels) in ORIGINAL pixel coordinates.
# ---------------------------------------------------------------------------


def _iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    tl = np.maximum(a[:, None, :2], b[:, :2])
    br = np.minimum(a[:, None, 2:], b[:, 2:])
    area_i = np.prod(br - tl, axis=2) * (tl < br).all(axis=2)
    area_a = np.prod(a[:, 2:] - a[:, :2], axis=1)
    area_b = np.prod(b[:, 2:] - b[:, :2], axis=1)
    return area_i / (area_a[:, None] + area_b - area_i + 1e-10)


def _crop_box_with_center_constraint(box: np.ndarray, crop: np.ndarray):
    cropped = box.copy()
    cropped[:, :2] = np.maximum(box[:, :2], crop[:2])
    cropped[:, 2:] = np.minimum(box[:, 2:], crop[2:])
    cropped[:, :2] -= crop[:2]
    cropped[:, 2:] -= crop[:2]
    centers = (box[:, :2] + box[:, 2:]) / 2
    valid = np.logical_and(crop[:2] <= centers, centers < crop[2:]).all(axis=1)
    valid = np.logical_and(valid, (cropped[:, :2] < cropped[:, 2:]).all(axis=1))
    return cropped, np.where(valid)[0]


def random_crop_iou(image, boxes, labels, aspect_ratio=(0.5, 2.0),
                    thresholds=(.0, .1, .3, .5, .7, .9), scaling=(.3, 1.0),
                    num_attempts=50, allow_no_crop=True, cover_all_box=False):
    """Paddle-faithful IoU-aware random crop. Returns (image, boxes, labels), possibly unchanged."""
    if len(boxes) == 0:
        return image, boxes, labels
    h, w = image.shape[:2]
    thresh_list = list(thresholds) + (["no_crop"] if allow_no_crop else [])
    random.shuffle(thresh_list)

    for thresh in thresh_list:
        if thresh == "no_crop":
            return image, boxes, labels
        for _ in range(num_attempts):
            scale = np.random.uniform(*scaling)
            min_ar, max_ar = aspect_ratio
            ar = np.random.uniform(max(min_ar, scale**2), min(max_ar, scale**-2))
            crop_h, crop_w = int(h * scale / np.sqrt(ar)), int(w * scale * np.sqrt(ar))
            if crop_h <= 0 or crop_w <= 0 or crop_h >= h or crop_w >= w:
                continue
            crop_y, crop_x = np.random.randint(0, h - crop_h), np.random.randint(0, w - crop_w)
            crop_box = np.array([crop_x, crop_y, crop_x + crop_w, crop_y + crop_h], dtype=np.float32)

            iou = _iou_matrix(boxes, crop_box[None, :])
            if iou.max() < thresh:
                continue
            if cover_all_box and iou.min() < thresh:
                continue
            cropped_boxes, valid_ids = _crop_box_with_center_constraint(boxes, crop_box)
            if valid_ids.size == 0:
                continue

            x1, y1, x2, y2 = crop_box.astype(int)
            return image[y1:y2, x1:x2], cropped_boxes[valid_ids], labels[valid_ids]
    return image, boxes, labels


# ---------------------------------------------------------------------------
# RandomDistort - ported from ppdet's RandomDistort, same defaults, same formulas.
# Operates on an RGB float32 image; each sub-effect has its own 50% chance.
# ---------------------------------------------------------------------------


def _apply_hue(img, low=-18, high=18, prob=0.5):
    if np.random.uniform(0., 1.) < prob:
        return img
    delta = np.random.uniform(low, high)
    u, w = np.cos(delta * np.pi), np.sin(delta * np.pi)
    bt = np.array([[1., 0., 0.], [0., u, -w], [0., w, u]])
    tyiq = np.array([[.299, .587, .114], [.596, -.274, -.321], [.211, -.523, .311]])
    ityiq = np.array([[1., .956, .621], [1., -.272, -.647], [1., -1.107, 1.705]])
    return img @ (ityiq @ bt @ tyiq).T


def _apply_saturation(img, low=0.5, high=1.5, prob=0.5):
    if np.random.uniform(0., 1.) < prob:
        return img
    delta = np.random.uniform(low, high)
    gray = (img * np.array([[[.299, .587, .114]]], dtype=np.float32)).sum(axis=2, keepdims=True)
    return img * delta + gray * (1.0 - delta)


def _apply_contrast(img, low=0.5, high=1.5, prob=0.5):
    if np.random.uniform(0., 1.) < prob:
        return img
    return img * np.random.uniform(low, high)


def _apply_brightness(img, low=0.5, high=1.5, prob=0.5):
    if np.random.uniform(0., 1.) < prob:
        return img
    return img + np.random.uniform(low, high)


def random_distort(img: np.ndarray) -> np.ndarray:
    """img: RGB, any numeric dtype -> returns float32 RGB, same value range as input."""
    img = img.astype(np.float32)
    funcs = [_apply_brightness, _apply_contrast, _apply_saturation, _apply_hue]
    random.shuffle(funcs)  # random_apply=True, count=4 in Paddle's defaults: all 4, random order
    for f in funcs:
        img = f(img)
    return img


# ---------------------------------------------------------------------------
# 1. Training transform: RandomCrop -> RandomFlip -> RandomDistort -> multiscale letterbox
# ---------------------------------------------------------------------------


class PICODETLetterboxTrainTransform:
    """Drop-in replacement for PICODETTrainTransform.

    Order matches Paddle's TrainReader: Decode(->RGB) is done by the dataset before this
    runs; RandomCrop, RandomFlip, RandomDistort happen here in original-image RGB space,
    then a (possibly downscaled, per MULTISCALE) letterbox resize into the fixed canvas.
    """

    _MEAN = np.array(IMAGENET_MEAN, dtype=np.float32)
    _STD = np.array(IMAGENET_STD, dtype=np.float32)

    def __init__(self, max_labels: int = 50, flip_prob: float = 0.5, hsv_prob: float = 0.0):
        self.max_labels = max_labels
        self.flip_prob = flip_prob
        self.random_crop = RANDOM_CROP
        self.random_distort = RANDOM_DISTORT
        self.multiscale = MULTISCALE
        self.multiscale_fractions = MULTISCALE_FRACTIONS
        # hsv_prob (LibreYOLO's own HSV jitter) is superseded by RandomDistort when that's on

    def __call__(self, image, targets, input_dim):
        boxes = targets[:, :4].copy()
        labels = targets[:, 4].copy()
        rgb = np.ascontiguousarray(image[:, :, ::-1])  # BGR -> RGB, matches Paddle's Decode

        if self.random_crop and len(boxes) > 0:
            rgb, boxes, labels = random_crop_iou(rgb, boxes, labels)

        rgb, boxes = mirror(rgb, boxes, self.flip_prob)

        rgb = rgb.astype(np.float32)
        if self.random_distort:
            rgb = random_distort(rgb)
        rgb = np.clip(rgb, 0, 255)

        target_h, target_w = _target_hw(input_dim)
        if self.multiscale:
            frac = random.choice(self.multiscale_fractions)
            sub_h, sub_w = max(32, int(target_h * frac)), max(32, int(target_w * frac))
        else:
            sub_h, sub_w = target_h, target_w

        sub_canvas, ratio, _, _ = letterbox_np(rgb, (sub_h, sub_w))
        canvas = np.full((target_h, target_w, 3), PAD_VALUE, dtype=np.float32)
        canvas[:sub_h, :sub_w] = sub_canvas  # top-left anchored, extra gray pad beyond sub-canvas

        canvas = (canvas - self._MEAN) / self._STD
        img_chw = np.ascontiguousarray(canvas.transpose(2, 0, 1), dtype=np.float32)

        padded_labels = np.zeros((self.max_labels, 5), dtype=np.float32)
        if len(boxes) > 0:
            from libreyolo.data.augment.boxes import xyxy2cxcywh
            b = xyxy2cxcywh(boxes * ratio)  # same ratio used for the sub-canvas -> top-left, no offset
            mask = np.minimum(b[:, 2], b[:, 3]) > 1
            b, lb = b[mask], labels[mask]
            n = min(len(b), self.max_labels)
            if n > 0:
                padded_labels[:n, 0] = lb[:n]
                padded_labels[:n, 1:] = b[:n]

        return img_chw, padded_labels


# ---------------------------------------------------------------------------
# 2. Validation-during-training (the mAP/AR train() reports each epoch) - fixed
#    canvas, no crop/distort/multiscale: eval must be deterministic.
# ---------------------------------------------------------------------------


class PICODETLetterboxValPreprocessor(BaseValPreprocessor):
    """PicoDet val preprocessor using letterbox instead of the default stretch resize."""

    _MEAN = np.array(IMAGENET_MEAN, dtype=np.float32).reshape(3, 1, 1)
    _STD = np.array(IMAGENET_STD, dtype=np.float32).reshape(3, 1, 1)

    @property
    def normalize(self) -> bool:
        return False  # custom_normalization=True below means the validator skips its own /255

    @property
    def custom_normalization(self) -> bool:
        return True

    @property
    def uses_letterbox(self) -> bool:
        return True

    @property
    def wants_unresized_image(self) -> bool:
        # Gives us the ORIGINAL image and ORIGINAL-PIXEL-SPACE targets (verified against
        # data/dataset.py: pull_item() explicitly undoes any dataset-level pre-letterbox
        # before handing off when this flag is set) -- so __call__ below must do its own
        # box transform, matching letterbox_scale() exactly.
        return True

    def letterbox_scale(self, orig_h: int, orig_w: int, imgsz) -> Tuple[float, float, float]:
        target_h, target_w = _target_hw(imgsz)
        r = min(target_h / orig_h, target_w / orig_w)
        return r, 0.0, 0.0  # top-left padding -> zero offset

    def __call__(self, img: np.ndarray, targets: np.ndarray, input_size) -> Tuple[np.ndarray, np.ndarray]:
        rgb = img[:, :, ::-1]  # BGR -> RGB
        canvas, ratio, _, _ = letterbox_np(rgb, input_size)
        chw = canvas.astype(np.float32).transpose(2, 0, 1)
        chw = (chw - self._MEAN) / self._STD

        padded = np.zeros((self.max_labels, 5), dtype=np.float32)
        if len(targets) > 0:
            t = np.array(targets, dtype=np.float32).copy()
            t[:, :4] *= ratio  # targets arrive in ORIGINAL pixel coords; see wants_unresized_image
            n = min(len(t), self.max_labels)
            padded[:n] = t[:n]
        return chw.astype(np.float32), padded


# ---------------------------------------------------------------------------
# 3. Manual inference (preprocess + un-letterbox postprocess)
# ---------------------------------------------------------------------------


def letterbox_preprocess_numpy(img_rgb_hwc: np.ndarray, input_size=320):
    canvas, ratio, new_h, new_w = letterbox_np(img_rgb_hwc, input_size)
    arr = canvas.astype(np.float32)
    arr -= np.array(IMAGENET_MEAN, dtype=np.float32)
    arr /= np.array(IMAGENET_STD, dtype=np.float32)
    return arr.transpose(2, 0, 1), ratio


def letterbox_predict(model, image, target_hw=(320, 320), conf_thres: float = 0.25,
                      iou_thres: float = 0.45, max_det: int = 100):
    """Run a LibrePICODET model with letterbox pre/postprocessing on one image.

    ``model`` is any LibrePICODET / PicoDetLCNet instance already built with the
    activation/gate it was trained with (this function does not touch that).
    """
    img = ImageLoader.load(image, color_format="auto")
    orig_w, orig_h = img.size
    chw, ratio = letterbox_preprocess_numpy(np.array(img), target_hw)
    tensor = torch.from_numpy(chw).unsqueeze(0).to(next(model.model.parameters()).device)

    model.model.eval()
    with torch.inference_mode():
        output = model.model(tensor)

    # original_size=None -> boxes come back in CANVAS pixel coords (no rescale applied),
    # so we can invert our own letterbox instead of the framework's stretch rescale.
    result = _picodet_decode(
        output, conf_thres=conf_thres, iou_thres=iou_thres,
        input_size=target_hw, original_size=None, max_det=max_det,
    )
    if result["num_detections"]:
        boxes = np.asarray(result["boxes"], dtype=np.float32) / ratio  # top-left pad -> no offset
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, orig_w)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, orig_h)
        result["boxes"] = boxes.tolist()
    return result


# ---------------------------------------------------------------------------
# Wire it in
# ---------------------------------------------------------------------------

_picodet_trainer.PICODETTrainTransform = PICODETLetterboxTrainTransform
_picodet_trainer.PICODETTrainer.create_transforms = lambda self: (
    PICODETLetterboxTrainTransform(max_labels=50, flip_prob=self.config.flip_prob,
                                   hsv_prob=self.config.hsv_prob),
    _picodet_trainer.MosaicMixupDataset,
)
LibrePICODET.val_preprocessor_class = PICODETLetterboxValPreprocessor
_picodet_utils.preprocess_numpy = letterbox_preprocess_numpy


# ---------------------------------------------------------------------------
# 4. THE ACTUAL BUG: PicoDet's own _postprocess() never forwarded the
#    `letterbox` flag the validator passes it, and postprocess.picodet.postprocess()
#    always un-scales predictions with a two-axis STRETCH formula regardless (see its
#    own comment: "PICODET uses simple resize, not letterbox"). Meanwhile the GT
#    reconstruction (PICODETLetterboxValPreprocessor.letterbox_scale, above) correctly
#    un-letterboxes. Predictions and GT were therefore being rescaled with two
#    DIFFERENT formulas every single validation step -- tanking IoU between them
#    regardless of how well the model was actually trained. This is very likely why
#    ground-truth boxes looked correct in visualize_letterbox.py while mAP/AR stayed
#    catastrophically low even with every augmentation toggle off.
# ---------------------------------------------------------------------------

def _letterbox_aware_postprocess(self, output, conf_thres, iou_thres, original_size,
                                 max_det=100, ratio=1.0, **kwargs):
    input_size = kwargs.get("input_size", self.input_size)
    target_h, target_w = _target_hw(input_size)
    orig_w, orig_h = original_size  # matches the framework's own (w, h) convention here

    # original_size=None -> canvas-space boxes, no rescale applied inside the decoder;
    # we invert our own letterbox instead of its stretch formula.
    result = _picodet_decode(
        output, conf_thres=conf_thres, iou_thres=iou_thres,
        input_size=input_size, original_size=None, max_det=max_det,
    )
    if result["num_detections"]:
        r = min(target_h / orig_h, target_w / orig_w)  # SAME formula as letterbox_scale/__call__
        boxes = np.asarray(result["boxes"], dtype=np.float32) / r  # top-left pad -> no offset
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, orig_w)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, orig_h)
        result["boxes"] = torch.from_numpy(boxes)
        result["scores"] = torch.as_tensor(result["scores"], dtype=torch.float32)
        result["classes"] = torch.as_tensor(result["classes"], dtype=torch.int64)
    return result


LibrePICODET._postprocess = _letterbox_aware_postprocess

