#!/usr/bin/env python3
"""Export the converted PicoDet PyTorch module back to ONNX and verify it.

    pip install torch onnx onnxruntime numpy
    python export_to_onnx.py --pt picodet_torch.pt --out picodet_from_torch.onnx
"""
import argparse

import numpy as np
import onnxruntime as ort
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pt", required=True)
    ap.add_argument("--out", default="picodet_from_torch.onnx")
    ap.add_argument("--size", type=int, default=320)
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--dynamic-batch", action="store_true", help="allow variable batch size")
    args = ap.parse_args()

    model = torch.load(args.pt, weights_only=False).eval()
    dummy = torch.randn(1, 3, args.size, args.size)

    dyn = {"image": {0: "batch"}, "boxes": {0: "batch"}, "scores": {0: "batch"}} if args.dynamic_batch else None
    torch.onnx.export(
        model, dummy, args.out,
        input_names=["image"], output_names=["boxes", "scores"],
        opset_version=args.opset, dynamic_axes=dyn,
        dynamo=False,
    )
    print(f"saved -> {args.out}")

    # verify against the PyTorch module
    x = np.random.RandomState(1).randn(1, 3, args.size, args.size).astype("float32")
    with torch.no_grad():
        ref = [o.numpy() for o in model(torch.from_numpy(x))]
    sess = ort.InferenceSession(args.out, providers=["CPUExecutionProvider"])
    out = sess.run(None, {"image": x})
    for name, a, b in zip(["boxes", "scores"], ref, out):
        print(f"{name}: shape {b.shape}, max abs diff vs PyTorch = {np.abs(a - b).max():.2e}")


if __name__ == "__main__":
    main()
