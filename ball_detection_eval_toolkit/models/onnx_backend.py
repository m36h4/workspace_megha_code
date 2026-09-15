"""
ONNX Runtime backend for ball detection.

IMPORTANT — this is the one piece that is model-specific and MUST be
confirmed against the actual exported model before trusting numbers:

    1. Output format: this code assumes the ONNX graph's raw output is
       ALREADY decoded to boxes + scores (i.e. NanoDet/PicoDet's
       post-processing / decode head is baked into the graph, which is
       the common export setting for both frameworks). If the export is
       "raw head output" (per-stride classification + distribution logits,
       undecoded), you need the model's `decode_output()` function instead
       of `_parse_output()` below -- these are architecture-specific
       (NanoDet: distribution focal loss regression; PicoDet: similar
       GFL-style head). Tell me which export mode you used and I'll swap
       this in.
    2. Preprocessing (normalization mean/std, RGB vs BGR, input layout)
       must match training exactly -- confirm against the model's config.

Assumed output tensor shape (common ONNX export convention for both
NanoDet-Plus and PicoDet when exported with decoding included):
    output: (1, N, 6) as [x1, y1, x2, y2, score, class_id]
    OR
    output: (1, N, 5) as [x1, y1, x2, y2, score]  (single class -> class dropped)

The parser below auto-detects which of these it received. If your model's
ONNX output differs (e.g. separate boxes/scores/labels tensors, as in some
NanoDet export scripts), see `_parse_output()` and adjust there --
that's the only function that needs to change.
"""

from typing import Optional, Tuple

import numpy as np
import onnxruntime as ort

from core.preprocess import letterbox_resize, normalize
from core.nms import nms_xyxy


class ONNXBallDetector:
    def __init__(self, onnx_path: str, input_size: Tuple[int, int] = (480, 320),
                 mean=(0.0, 0.0, 0.0), std=(255.0, 255.0, 255.0),
                 score_thresh: float = 0.05, nms_iou_thresh: float = 0.5,
                 providers: Optional[list] = None):
        """
        input_size: (width, height) -- per client spec, 480x320 (W x H). Confirm
                    orientation against the actual model input tensor shape.
        mean/std: normalization applied as (pixel - mean) / std, RGB order.
                  Defaults to simple 0-1 scaling; override for ImageNet stats
                  ([123.675, 116.28, 103.53], [58.395, 57.12, 57.375]) if that's
                  what training used.
        """
        self.input_w, self.input_h = input_size
        self.mean = mean
        self.std = std
        self.score_thresh = score_thresh
        self.nms_iou_thresh = nms_iou_thresh

        avail = ort.get_available_providers()
        if providers is None:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        providers = [p for p in providers if p in avail] or ["CPUExecutionProvider"]

        self.session = ort.InferenceSession(onnx_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]

        declared_shape = self.session.get_inputs()[0].shape
        print(f"[ONNXBallDetector] loaded {onnx_path}")
        print(f"  input: {self.input_name}, declared shape: {declared_shape}")
        print(f"  outputs: {self.output_names}")
        print(f"  using providers: {providers}")

    def _preprocess(self, image_rgb: np.ndarray):
        padded, transform = letterbox_resize(image_rgb, self.input_w, self.input_h)
        chw = normalize(padded, self.mean, self.std, to_chw=True)
        batch = chw[None, ...].astype(np.float32)
        return batch, transform

    def _parse_output(self, raw_outputs) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns (boxes_xyxy, scores) in MODEL INPUT pixel space (pre-letterbox-inverse).
        Adjust this function to match your actual export -- see module docstring.
        """
        out = raw_outputs[0]
        out = np.array(out)

        if out.ndim == 3:
            out = out[0]  # drop batch dim -> (N, C)

        if out.shape[-1] == 0 or out.shape[0] == 0:
            return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32)

        if out.shape[-1] >= 5:
            boxes = out[:, 0:4].astype(np.float32)
            scores = out[:, 4].astype(np.float32)
            return boxes, scores

        raise ValueError(
            f"Unrecognized ONNX output shape {out.shape}; "
            f"update ONNXBallDetector._parse_output() for this export format."
        )

    def predict(self, image_rgb: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns (boxes_xyxy, scores) in ORIGINAL image pixel coordinates,
        after NMS and score thresholding.
        """
        batch, transform = self._preprocess(image_rgb)
        raw = self.session.run(self.output_names, {self.input_name: batch})
        boxes_model, scores = self._parse_output(raw)

        boxes_orig = transform.to_orig_xyxy(boxes_model)
        boxes_orig, scores = nms_xyxy(boxes_orig, scores,
                                       iou_thresh=self.nms_iou_thresh,
                                       score_thresh=self.score_thresh)
        return boxes_orig, scores
