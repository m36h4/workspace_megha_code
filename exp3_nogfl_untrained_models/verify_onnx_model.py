"""Checks an exported PicoDet ONNX before it is shared. Runs on your machine, no dataset needed.

    python verify_onnx_model.py --onnx exp3_nogfl_untrained.onnx --weights exp3_nogfl_untrained.pt \
        --act leakyrelu --gate sigmoid
    python verify_onnx_model.py --self-test          # proves the decode math itself (see 3 below)

What it checks
  1. Interface: input name / shape, output name / shape, opset, single file (no .data).
  2. Ops: lists every op type; FAIL on ops the client cannot compile (HardSwish, NMS, MatMul,
     Softmax in a no-GFL model); notes HardSigmoid only in the x*hardsigmoid(x) pattern.
  3. Numbers: the ONNX output is compared with an INDEPENDENT NumPy decode of the PyTorch
     model's raw head outputs (written separately from the exporter's decode). With --weights,
     it uses your checkpoint; untrained weights give near-zero boxes, so --self-test repeats the
     check on a random model with large, non-zero outputs, exported through the same exporter.
"""
import argparse
import collections
import os
import subprocess
import sys
import tempfile

import numpy as np
import onnx
import onnxruntime as ort
import torch

BANNED = {"HardSwish", "NonMaxSuppression", "MatMul", "Gemm", "TopK"}


def interface_and_ops(path, no_gfl=None):
    m = onnx.load(path)
    ok = True
    i, o = m.graph.input[0], m.graph.output
    dims = lambda v: [d.dim_value or d.dim_param for d in v.type.tensor_type.shape.dim]
    print(f"input : name='{i.name}'  shape={dims(i)}")
    for out in o:
        print(f"output: name='{out.name}' shape={dims(out)}")
    print("opset :", [(x.domain or 'ai.onnx', x.version) for x in m.opset_import])
    ext = any(t.data_location == onnx.TensorProto.EXTERNAL for t in m.graph.initializer)
    print("single file:", "yes" if not ext and not os.path.exists(path + ".data") else "NO (external .data)")
    ok &= not ext
    c = collections.Counter(n.op_type for n in m.graph.node)
    print("ops   :", dict(sorted(c.items(), key=lambda kv: -kv[1])))
    bad = sorted(set(c) & BANNED)
    if bad:
        print("FAIL  banned ops:", bad); ok = False
    if no_gfl and ("Softmax" in c or "ReduceSum" in c):
        print("FAIL  no-GFL model still contains the GFL decode (Softmax/ReduceSum)"); ok = False
    if "HardSigmoid" in c:
        muls = {frozenset(n.input) for n in m.graph.node if n.op_type == "Mul"}
        hs = [n for n in m.graph.node if n.op_type == "HardSigmoid"]
        good = sum(frozenset((n.input[0], n.output[0])) in muls for n in hs)
        print(f"note  HardSigmoid x{len(hs)}: {good} in the x*hardsigmoid(x) pattern"
              + ("" if good == len(hs) else "  <-- FAIL, others are not"))
        ok &= good == len(hs)
    return ok, i.name, dims(i)


def numpy_decode(cls_list, box_list, strides, reg_max):
    """Independent reference: raw head maps -> (N, 4+nc) [x1,y1,x2,y2 | sigmoid score]."""
    rows = []
    for c, b, s in zip(cls_list, box_list, strides):
        c, b = c[0], b[0]
        h, w = c.shape[1:]
        if reg_max == 0:
            d = np.maximum(b, 0.0) * s                                       # direct distances
        else:
            b = b.reshape(4, reg_max + 1, h, w)
            e = np.exp(b - b.max(axis=1, keepdims=True))
            d = (e / e.sum(axis=1, keepdims=True) * np.arange(reg_max + 1)[None, :, None, None]).sum(1) * s
        ys, xs = np.meshgrid((np.arange(h) + 0.5) * s, (np.arange(w) + 0.5) * s, indexing="ij")
        box = np.stack([xs - d[0], ys - d[1], xs + d[2], ys + d[3]]).reshape(4, -1)
        sc = 1.0 / (1.0 + np.exp(-c.reshape(c.shape[0], -1)))
        rows.append(np.concatenate([box, sc]).T)
    return np.concatenate(rows)


def compare(onnx_path, net, input_name, h, w, label):
    x = np.random.RandomState(0).randn(1, 3, h, w).astype(np.float32)
    head = net.head
    head.export = False
    with torch.no_grad():
        cls_l, box_l = net(torch.from_numpy(x))
    ref = numpy_decode([t.numpy() for t in cls_l], [t.numpy() for t in box_l], tuple(head.strides), head.reg_max)
    got = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"]).run(None, {input_name: x})[0]
    got = got[0] if got.shape[1] == ref.shape[0] else got[0].T          # accept (1,N,C) or (1,C,N)
    diff = np.abs(got - ref).max()
    span = np.abs(ref[:, 2] - ref[:, 0]).max()
    ok = got.shape == ref.shape and diff < 1e-3
    print(f"{label}: ONNX vs independent NumPy decode  shape {got.shape}  max abs diff {diff:.2e}  "
          f"(largest box width {span:.1f}px)  -> {'OK' if ok else 'FAIL'}")
    if span < 1.0:
        print("   note: boxes are ~0 px (untrained weights), so this check is weak; run --self-test as well")
    return ok


def build(preset, act, gate, reg_max, weights=None):
    from picodet_activation import set_activation
    set_activation(act, gate=gate)
    from picodet_esnet_tiny import PicoDetTiny, PRESETS, load_tiny
    if weights:
        return load_tiny(weights, preset=preset).model.eval().cpu()
    p = PRESETS[preset]
    return PicoDetTiny(size="s", nb_classes=1, scale=p["scale"], s16=p["s16"], neck=p["neck"], k=p["k"],
                       reg_max=reg_max).model.eval()


def self_test(preset, act, gate, h, w):
    ok = True
    here = os.path.dirname(os.path.abspath(__file__))
    for reg_max in (0, 7):
        torch.manual_seed(0)
        net = build(preset, act, gate, reg_max)
        for m in net.modules():                                   # make outputs large and non-zero
            if isinstance(m, torch.nn.BatchNorm2d):
                m.running_mean.normal_(0, 0.1); m.running_var.uniform_(0.5, 1.5)
        nc = net.head.num_classes
        with torch.no_grad():
            for last in net.head.gfl_cls:                         # every pyramid level
                torch.nn.init.normal_(last.weight, 0, 0.5)
                torch.nn.init.normal_(last.bias, 0, 1.0)
                if reg_max == 0:                                  # keep ReLU active: real box widths
                    last.bias[nc:].uniform_(1.0, 3.0)
        with tempfile.TemporaryDirectory() as d:
            pt, ox = os.path.join(d, "m.pt"), os.path.join(d, "m.onnx")
            torch.save({"model": net.state_dict(), "nc": 1}, pt)
            r = subprocess.run([sys.executable, os.path.join(here, "export_onnx.py"), "--weights", pt,
                                "--arch", "esnet_tiny", "--tiny-preset", preset, "--act", act, "--gate", gate,
                                "--imgsz", str(h), str(w), "--out", ox], capture_output=True, text=True)
            if not os.path.exists(ox):
                print(r.stdout[-2000:], r.stderr[-2000:]); return False
            name = onnx.load(ox).graph.input[0].name
            ok &= compare(ox, build(preset, act, gate, reg_max, weights=pt), name, h, w,
                          f"self-test {'no-GFL' if reg_max == 0 else 'GFL   '}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx")
    ap.add_argument("--weights", help=".pt the ONNX was exported from (enables the numeric check)")
    ap.add_argument("--preset", default="t1")
    ap.add_argument("--act", default="leakyrelu")
    ap.add_argument("--gate", default="sigmoid")
    ap.add_argument("--expect-input-name", default="input")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    ok = True
    if a.onnx:
        no_gfl = None
        if a.weights:
            from picodet_esnet_tiny import _reg_max_from_checkpoint
            no_gfl = _reg_max_from_checkpoint(a.weights) == 0
            print("checkpoint head:", "no-GFL (4 direct distances)" if no_gfl else "GFL (4 x 8 distribution)")
        good, name, dims = interface_and_ops(a.onnx, no_gfl)
        ok &= good
        if name != a.expect_input_name:
            print(f"FAIL  input name is '{name}', expected '{a.expect_input_name}'"); ok = False
        if a.weights:
            h, w = int(dims[2]), int(dims[3])
            ok &= compare(a.onnx, build(a.preset, a.act, a.gate, 0, weights=a.weights), name, h, w, "your file")
    if a.self_test:
        ok &= self_test(a.preset, a.act, a.gate, 320, 480)
    print("\nRESULT:", "ALL CHECKS PASSED" if ok else "SOMETHING FAILED (see above)")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
