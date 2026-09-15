"""
NanoDet-Plus (.pt) inference backend.

*** STATUS: SCAFFOLDED, NOT YET FUNCTIONAL ***

This backend needs three things from you that I don't have yet:
    1. The NanoDet-Plus repo's model-building code (or a pip-installable
       package) to reconstruct the nn.Module architecture from the .pt
       checkpoint's state_dict -- a raw state_dict alone has no attached
       class/architecture.
    2. The exact training config .yml (backbone variant, `reg_max`, feature
       map strides, number of classes, input size) -- NanoDet's head uses
       Generalized Focal Loss with a distribution-based box regression
       (integral of a discrete distribution over `reg_max+1` bins per side),
       which cannot be decoded correctly without these values.
    3. Confirmation of preprocessing used at training time (mean/std, color
       order) -- NanoDet-Plus's default configs commonly use ImageNet-style
       mean=[103.53, 116.28, 123.675], std=[57.375, 57.12, 58.395] in BGR
       order, but this varies by config and MUST be confirmed, not assumed.

Once you provide these, this file's `load_model()` and `decode_output()`
are the only two functions that need real implementations -- everything
else in the toolkit (dataset loading, letterbox preprocessing inverse
transform, NMS, metrics, reporting) already works identically for this
backend as it does for ONNX, since PerImagePrediction / evaluate() /
the CLI wiring are all backend-agnostic.

Sketch of what `decode_output` will look like once the config is known
(standard NanoDet-Plus GFL decode, for reference / review):

    for each feature level (stride s in strides):
        cls_score, bbox_pred = head_outputs[level]   # (H*W, num_classes), (H*W, 4*(reg_max+1))
        bbox_pred = bbox_pred.reshape(-1, 4, reg_max + 1)
        bbox_pred = softmax(bbox_pred, dim=-1) @ arange(reg_max + 1)   # integral -> distances
        # bbox_pred is now (H*W, 4) = distances [left, top, right, bottom] from each anchor point
        points = generate_grid_points(H, W, stride=s)
        x1 = points_x - bbox_pred[:, 0] * s
        y1 = points_y - bbox_pred[:, 1] * s
        x2 = points_x + bbox_pred[:, 2] * s
        y2 = points_y + bbox_pred[:, 3] * s
        scores = sigmoid(cls_score).max(dim=-1)
    concatenate across levels, then NMS.

This is a best-effort sketch from the public NanoDet-Plus architecture and
MUST be verified against your actual training config before trusting any
numbers it produces.
"""

from typing import Tuple

import numpy as np

from core.preprocess import letterbox_resize, normalize
from core.nms import nms_xyxy


class NanoDetBallDetector:
    def __init__(self, checkpoint_path: str, config_path: str,
                 input_size: Tuple[int, int] = (480, 320),
                 score_thresh: float = 0.05, nms_iou_thresh: float = 0.5,
                 device: str = "cpu"):
        """
        checkpoint_path: .pt file
        config_path: NanoDet-Plus training .yml -- REQUIRED, not optional,
                     because reg_max/strides/backbone are read from it.
        """
        self.input_w, self.input_h = input_size
        self.score_thresh = score_thresh
        self.nms_iou_thresh = nms_iou_thresh
        self.device = device

        self.config = self._load_config(config_path)
        self.model = self._load_model(checkpoint_path, self.config)

        # Defaults match NanoDet-Plus's common ImageNet-style config; OVERRIDE
        # via self.config if the actual training config differs.
        self.mean = self.config.get("mean", [103.53, 116.28, 123.675])
        self.std = self.config.get("std", [57.375, 57.12, 58.395])
        self.bgr = self.config.get("bgr", True)

    def _load_config(self, config_path: str) -> dict:
        import yaml
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
        return cfg

    def _load_model(self, checkpoint_path: str, config: dict):
        """
        *** NOT YET IMPLEMENTED ***
        Needs the NanoDet-Plus repo (github.com/RangiLyu/nanodet) importable,
        e.g.:
            from nanodet.model.arch import build_model
            model = build_model(config["model"])
            ckpt = torch.load(checkpoint_path, map_location="cpu")
            state_dict = ckpt.get("state_dict", ckpt)
            # strip "model." / "avg_model." prefixes if present, per NanoDet's
            # own checkpoint saving convention
            model.load_state_dict(state_dict, strict=True)
            model.eval()
            return model
        Send me either:
          (a) the nanodet package/repo pinned at the commit you trained with, or
          (b) confirmation you're fine adding `pip install nanodet` as a dependency,
        and I'll complete this.
        """
        raise NotImplementedError(
            "NanoDetBallDetector needs the NanoDet-Plus model-building code "
            "(the repo or package used for training) to reconstruct the "
            "architecture before loading the .pt state_dict. See this "
            "method's docstring for what to send me."
        )

    def _preprocess(self, image_rgb: np.ndarray):
        padded, transform = letterbox_resize(image_rgb, self.input_w, self.input_h)
        img = padded[:, :, ::-1] if self.bgr else padded  # RGB -> BGR if needed
        chw = normalize(img, self.mean, self.std, to_chw=True)
        return chw[None, ...].astype(np.float32), transform

    def decode_output(self, raw_head_outputs) -> Tuple[np.ndarray, np.ndarray]:
        """
        *** NOT YET IMPLEMENTED *** -- see module docstring for the decode sketch.
        Returns (boxes_xyxy, scores) in MODEL INPUT pixel space.
        """
        raise NotImplementedError(
            "decode_output needs reg_max and strides from the training config "
            "to correctly integrate the GFL distribution head into box "
            "coordinates. See module docstring for the algorithm sketch."
        )

    def predict(self, image_rgb: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        import torch

        batch, transform = self._preprocess(image_rgb)
        with torch.no_grad():
            raw = self.model(torch.from_numpy(batch).to(self.device))
        boxes_model, scores = self.decode_output(raw)

        boxes_orig = transform.to_orig_xyxy(boxes_model)
        boxes_orig, scores = nms_xyxy(boxes_orig, scores,
                                       iou_thresh=self.nms_iou_thresh,
                                       score_thresh=self.score_thresh)
        return boxes_orig, scores
