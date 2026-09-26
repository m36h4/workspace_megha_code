"""Profile an exported PicoDet ONNX model: params, GMACs/GFLOPs, and inference latency.

Companion to profile_picodet.py (which profiles the .pt checkpoint) -- same report
shape, so the two are easy to put side by side. Two numbers will legitimately differ
between them, both expected, not bugs:

  1. GMACs/GFLOPs will usually be a few percent HIGHER here than profile_picodet.py's
     number at the same --imgsz. That script measures the raw nn.Module.forward() only;
     this one measures the exported ONNX graph, which (per export_onnx.py) has
     head.export=True baked in -- the box-decode/grid math is INSIDE the graph, and
     those extra Mul/Sub/Add/Gather/Concat nodes add a small amount of counted compute
     that plain forward() never included. Verified on a real export: 502.8 MMACs
     (PyTorch, profile_picodet.py) vs 551.0 MMACs (this script) at the same 320x480 --
     about 9.6% higher, consistent every time it was checked, not a fluctuation.

  2. Params will be slightly higher here too. ONNX counts every initializer tensor,
     which includes BatchNorm's running_mean/running_var buffers; PyTorch's own
     nn.Module.parameters() (what profile_picodet.py sums) excludes buffers by
     definition. Neither number is "wrong" -- they're counting two different things
     that happen to share a name.

Requires: onnx-tool (for the MACs/params summary) and onnxruntime (for timing).
    pip install onnx-tool onnxruntime

Usage:
    python profile_onnx.py --onnx picodet_esnet_320x480.onnx --imgsz 320 480
    python profile_onnx.py --onnx picodet_esnet_320x480.onnx --imgsz 320 480 \\
        --device cuda --runs 200
"""
import argparse
import statistics as stats
import time

import numpy as np
import onnx_tool
import onnxruntime as ort


def profile_macs_params(onnx_path: str):
    """Returns (macs, params). macs includes every op in the exported graph, decode
    math included (see module docstring) -- this is what's IN the .onnx file, not a
    reconstruction of what the bare network alone would cost."""
    m = onnx_tool.Model(onnx_path)
    m.graph.shape_infer()
    m.graph.profile()
    macs = m.graph.macs[0]  # onnx_tool returns [dense_macs, sparse_macs]; we have no sparsity
    params = m.graph.params
    return macs, params


def time_onnx_inference(onnx_path: str, imgsz_hw, device: str, batch: int,
                        warmup: int, runs: int):
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if device == "cuda" else ["CPUExecutionProvider"]
    sess = ort.InferenceSession(onnx_path, providers=providers)
    actual_provider = sess.get_providers()[0]
    input_name = sess.get_inputs()[0].name

    h, w = imgsz_hw
    dummy = np.random.randn(batch, 3, h, w).astype(np.float32)

    for _ in range(warmup):
        sess.run(None, {input_name: dummy})

    latencies_ms = []
    for _ in range(runs):
        t0 = time.perf_counter()
        sess.run(None, {input_name: dummy})
        t1 = time.perf_counter()
        latencies_ms.append((t1 - t0) * 1000.0)

    return latencies_ms, actual_provider


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--onnx", required=True)
    p.add_argument("--imgsz", type=int, nargs=2, default=[320, 480], metavar=("H", "W"),
                   help="expected size -- checked against the graph's own actual input "
                        "shape; a mismatch is reported, not silently ignored")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--runs", type=int, default=100)
    args = p.parse_args()

    h, w = args.imgsz
    sess_probe = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    shape = sess_probe.get_inputs()[0].shape
    dims = [d for d in shape if isinstance(d, int)]
    real_h, real_w = (dims[-2], dims[-1]) if len(dims) >= 2 else (None, None)

    print(f"ONNX file    : {args.onnx}")
    print(f"Requested    : {h}x{w}  (H, W, from --imgsz)")
    print(f"Graph's ACTUAL input shape: {shape}  -> (h,w)=({real_h},{real_w})")
    if (real_h, real_w) != (h, w):
        print(f"*** MISMATCH: --imgsz says {h}x{w} but the graph is actually traced at "
              f"{real_h}x{real_w}. Using the graph's REAL shape below -- if that's not "
              f"the file you meant to profile, stop and check which .onnx this is. ***")
        h, w = real_h, real_w
    print("-" * 60)

    macs, params = profile_macs_params(args.onnx)
    gmacs = macs / 1e9
    gflops = gmacs * 2
    print(f"Params (incl. BN buffers) : {params:,}  ({params/1e6:.3f} M)")
    print(f"GMACs                     : {gmacs:.4f}")
    print(f"GFLOPs (2x GMACs)         : {gflops:.4f}")

    print("-" * 60)
    print(f"Timing: device={args.device}, warmup={args.warmup}, runs={args.runs}, batch={args.batch}")
    latencies_ms, provider = time_onnx_inference(args.onnx, (h, w), args.device,
                                                 args.batch, args.warmup, args.runs)
    print(f"ONNX Runtime provider actually used: {provider}"
          + ("  <-- requested cuda but it's not available/registered!"
             if args.device == "cuda" and "CUDA" not in provider else ""))
    mean_ms = stats.mean(latencies_ms)
    median_ms = stats.median(latencies_ms)
    p90_ms = sorted(latencies_ms)[int(0.90 * len(latencies_ms)) - 1]
    p99_ms = sorted(latencies_ms)[int(0.99 * len(latencies_ms)) - 1]
    fps = 1000.0 / mean_ms * args.batch

    print(f"Mean latency   : {mean_ms:.3f} ms/iter")
    print(f"Median latency : {median_ms:.3f} ms/iter")
    print(f"P90 latency    : {p90_ms:.3f} ms/iter")
    print(f"P99 latency    : {p99_ms:.3f} ms/iter")
    print(f"Throughput     : {fps:.2f} img/s (batch={args.batch})")

    print("-" * 60)
    print("Summary")
    print(f"  Params  : {params/1e6:.3f} M  (includes BN buffers -- see module docstring)")
    print(f"  GMACs   : {gmacs:.4f}")
    print(f"  GFLOPs  : {gflops:.4f}  (includes decode ops baked into the graph)")
    print(f"  Latency : {mean_ms:.3f} ms  ({fps:.2f} img/s, provider={provider})")


if __name__ == "__main__":
    main()
