#!/usr/bin/env python3
"""Convert a PicoDet ONNX file to a PyTorch module and verify it against onnxruntime.

    pip install torch onnx onnx2torch onnxruntime numpy
    python convert_to_torch.py --onnx primitive_picodet_Sim.onnx --out picodet_torch.pt
"""
import argparse

import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnx import numpy_helper
from onnx2torch import convert


def constants_to_initializers(model):
    """This export stores every weight as a Constant node; onnx2torch expects initializers."""
    keep = []
    for node in model.graph.node:
        if node.op_type == "Constant" and len(node.attribute) == 1 and node.attribute[0].name == "value":
            tensor = numpy_helper.to_array(node.attribute[0].t)
            model.graph.initializer.append(numpy_helper.from_array(tensor, name=node.output[0]))
        else:
            keep.append(node)
    del model.graph.node[:]
    model.graph.node.extend(keep)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--out", default="picodet_torch.pt", help="full-module file (torch.save)")
    ap.add_argument("--size", type=int, default=320)
    args = ap.parse_args()

    model = convert(constants_to_initializers(onnx.load(args.onnx))).eval()

    x = np.random.RandomState(0).randn(1, 3, args.size, args.size).astype("float32")

    sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    ref = sess.run(None, {sess.get_inputs()[0].name: x})
    with torch.no_grad():
        out = model(torch.from_numpy(x))
    out = [o.numpy() for o in (out if isinstance(out, (tuple, list)) else [out])]

    for i, (a, b) in enumerate(zip(ref, out)):
        print(f"output {i}: shape {b.shape}, max abs diff vs onnxruntime = {np.abs(a - b).max():.2e}")

    torch.save(model, args.out)
    print(f"saved -> {args.out}   (load with torch.load(path, weights_only=False))")


if __name__ == "__main__":
    main()
