"""
PicoDet (.pdparams) inference backend.

*** STATUS: SCAFFOLDED, NOT YET FUNCTIONAL ***

Same situation as NanoDet: a raw .pdparams file is just weight tensors --
it has no attached architecture. To run inference we need ONE of:

    Option A (recommended, discussed in README):
        Export PicoDet to ONNX once (requires `paddledet`/PaddleDetection
        installed for that one-time conversion script, using their own
        `tools/export_model.py` + the ONNX export step they document).
        After that, the model runs through `models/onnx_backend.py`
        exactly like everything else -- no Paddle dependency at inference
        time. This file would then not be needed at all.

    Option B (this file, if you'd rather not do a one-time ONNX export):
        Use `paddle.inference` (Paddle's lighter C++/Python inference API)
        against an exported `.pdmodel` + `.pdiparams` pair -- NOT the raw
        training `.pdparams` checkpoint. PaddleDetection's export script
        (`tools/export_model.py`) produces this pair from a training
        checkpoint + its config. This still requires paddledet at export
        time (once), but only `paddlepaddle` (not paddledet) at inference
        time, which is the "minimal deps" version of option B.

Either way, I need:
    1. The PicoDet training config .yml (paired with the .pdparams)
    2. Confirmation of preprocessing (PicoDet configs typically use
       mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], RGB --
       but this must be read from your actual config, not assumed)
    3. Your choice of Option A vs B above

This file currently implements the Option B path assuming an already
-exported `.pdmodel`/`.pdiparams` pair exists (produced by PaddleDetection's
export tool, which I can also script for you once I have the config).
"""

from typing import Tuple

import numpy as np

from core.preprocess import letterbox_resize, normalize
from core.nms import nms_xyxy


class PicoDetBallDetector:
    def __init__(self, model_dir: str, input_size: Tuple[int, int] = (480, 320),
                 mean=(123.675, 116.28, 103.53), std=(58.395, 57.12, 57.375),
                 score_thresh: float = 0.05, nms_iou_thresh: float = 0.5,
                 use_gpu: bool = False):
        """
        model_dir: directory containing `model.pdmodel` + `model.pdiparams`,
                   i.e. the OUTPUT of PaddleDetection's export_model.py --
                   NOT the raw training .pdparams checkpoint. See module
                   docstring: the raw checkpoint has no attached graph.

        If post-processing (NMS + decoding) is baked into the exported graph
        (PaddleDetection's default export usually includes this, similar to
        the ONNX backend's assumption), `_parse_output` below should work
        largely unchanged from the ONNX backend's logic. If you exported
        with `--exclude_nms` or similar, raw per-stride head outputs will
        need PicoDet-specific decoding (GFL-style, same family as NanoDet --
        see nanodet_backend.py's docstring for the general shape of that math).
        """
        self.input_w, self.input_h = input_size
        self.mean = mean
        self.std = std
        self.score_thresh = score_thresh
        self.nms_iou_thresh = nms_iou_thresh

        self._load_model(model_dir, use_gpu)

    def _load_model(self, model_dir: str, use_gpu: bool):
        """
        Requires `paddlepaddle` installed (not paddledet, per client's
        minimal-deps preference -- paddledet is only needed once, upstream,
        to produce model.pdmodel/model.pdiparams from the training checkpoint).
        """
        try:
            import paddle.inference as paddle_infer
        except ImportError:
            raise ImportError(
                "PicoDetBallDetector requires `paddlepaddle` "
                "(pip install paddlepaddle --break-system-packages for CPU, "
                "or paddlepaddle-gpu for GPU). paddledet itself is NOT needed "
                "here -- only for the one-time export step upstream."
            )
        from pathlib import Path
        model_dir = Path(model_dir)
        model_file = model_dir / "model.pdmodel"
        params_file = model_dir / "model.pdiparams"
        if not model_file.exists() or not params_file.exists():
            raise FileNotFoundError(
                f"Expected {model_file} and {params_file}. "
                f"If you only have a raw .pdparams training checkpoint, it "
                f"needs to go through PaddleDetection's export_model.py first "
                f"-- see this file's module docstring, Option A/B."
            )

        config = paddle_infer.Config(str(model_file), str(params_file))
        if use_gpu:
            config.enable_use_gpu(500, 0)
        else:
            config.disable_gpu()
        config.enable_memory_optim()
        self.predictor = paddle_infer.create_predictor(config)

        self.input_names = self.predictor.get_input_names()
        self.output_names = self.predictor.get_output_names()
        print(f"[PicoDetBallDetector] loaded {model_dir}")
        print(f"  inputs: {self.input_names}")
        print(f"  outputs: {self.output_names}")

    def _preprocess(self, image_rgb: np.ndarray):
        padded, transform = letterbox_resize(image_rgb, self.input_w, self.input_h)
        chw = normalize(padded, self.mean, self.std, to_chw=True)
        return chw[None, ...].astype(np.float32), transform

    def _parse_output(self, raw_output: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Assumes decoded output shaped (N, 6) as [class_id, score, x1, y1, x2, y2]
        -- this is PaddleDetection's typical export convention for detection
        models (note: DIFFERENT column order than the ONNX backend's assumed
        [x1,y1,x2,y2,score,class]) -- MUST be verified against your actual
        exported graph's output signature once available.
        """
        out = np.array(raw_output)
        if out.ndim == 3:
            out = out[0]
        if out.shape[0] == 0:
            return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32)

        if out.shape[-1] == 6:
            scores = out[:, 1].astype(np.float32)
            boxes = out[:, 2:6].astype(np.float32)
            return boxes, scores

        raise ValueError(
            f"Unrecognized PicoDet output shape {out.shape}; "
            f"update PicoDetBallDetector._parse_output() to match the actual "
            f"exported graph's output signature."
        )

    def predict(self, image_rgb: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        batch, transform = self._preprocess(image_rgb)

        input_handle = self.predictor.get_input_handle(self.input_names[0])
        input_handle.reshape(batch.shape)
        input_handle.copy_from_cpu(batch)

        # PaddleDetection-exported models commonly also need an "im_shape" /
        # "scale_factor" input for internal post-processing -- if
        # `len(self.input_names) > 1`, this needs to be supplied too. Flagging
        # here rather than guessing, since the exact extra input names/values
        # vary by export config.
        if len(self.input_names) > 1:
            raise NotImplementedError(
                f"Model expects additional inputs {self.input_names[1:]} "
                f"(commonly im_shape/scale_factor for PaddleDetection exports). "
                f"Wire these up once the export config is known."
            )

        self.predictor.run()
        output_handle = self.predictor.get_output_handle(self.output_names[0])
        raw_output = output_handle.copy_to_cpu()

        boxes_model, scores = self._parse_output(raw_output)
        boxes_orig = transform.to_orig_xyxy(boxes_model)
        boxes_orig, scores = nms_xyxy(boxes_orig, scores,
                                       iou_thresh=self.nms_iou_thresh,
                                       score_thresh=self.score_thresh)
        return boxes_orig, scores
