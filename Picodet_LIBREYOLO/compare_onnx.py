"""Compare multiple ONNX models (PicoDet-ESNet, PicoDet-LCNet, NanoDet-Plus, anything)
side by side: params, GMACs/GFLOPs, and latency, in one table.

Reuses profile_onnx.py's own functions -- same numbers you'd get running it on each
file separately, just tabulated. Keep profile_onnx.py in the same folder.

Each model's input shape (H, W) is read directly from its own graph, NOT from a shared
--imgsz flag. This is deliberate: your exports are traced at a fixed size (only the
batch axis is ever dynamic, via export_onnx.py's --dynamic-batch), so forcing a shared
--imgsz across models exported at different sizes would either crash on a shape
mismatch or (worse) silently compare them unfairly. Reading each graph's own shape
means you can compare models at different native resolutions and the table will show
you that difference honestly instead of hiding it.

Usage:
    python compare_onnx.py --onnx picodet_esnet_320x480.onnx picodet_lcnet_320x480.onnx nanodet_plus.onnx
    python compare_onnx.py --onnx a.onnx b.onnx --labels ESNet LCNet --runs 200
"""
import argparse

import onnxruntime as ort

from profile_onnx import profile_macs_params, time_onnx_inference


def get_native_hw(onnx_path: str):
    """Reads the model's own input shape. Returns (h, w); non-int dims (a dynamic
    batch axis, named e.g. 'batch') are assumed to be the batch dim and skipped --
    H and W are expected to be static ints, per export_onnx.py's contract."""
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    shape = sess.get_inputs()[0].shape  # e.g. [1, 3, 320, 480] or ['batch', 3, 320, 480]
    dims = [d for d in shape if isinstance(d, int)]
    if len(dims) < 3:
        raise ValueError(f"{onnx_path}: couldn't find a static (C,H,W) in input shape {shape}; "
                         f"this model may have more dynamic axes than expected.")
    # last two static dims are H, W (channel is also static but comes before them)
    h, w = dims[-2], dims[-1]
    return h, w


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--onnx", nargs="+", required=True)
    p.add_argument("--labels", nargs="+", default=None,
                   help="one label per --onnx path, same order (default: filename)")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--runs", type=int, default=100)
    args = p.parse_args()

    labels = args.labels or [p.rsplit("/", 1)[-1] for p in args.onnx]
    if len(labels) != len(args.onnx):
        raise SystemExit(f"--labels has {len(labels)} entries but --onnx has {len(args.onnx)}")

    rows = []
    for path, label in zip(args.onnx, labels):
        h, w = get_native_hw(path)
        macs, params = profile_macs_params(path)
        latencies_ms, provider = time_onnx_inference(path, (h, w), args.device,
                                                      args.batch, args.warmup, args.runs)
        mean_ms = sum(latencies_ms) / len(latencies_ms)
        fps = 1000.0 / mean_ms * args.batch
        rows.append({
            "label": label, "hw": f"{h}x{w}", "params_m": params / 1e6,
            "gmacs": macs / 1e9, "gflops": 2 * macs / 1e9,
            "latency_ms": mean_ms, "fps": fps, "provider": provider,
        })
        print(f"done: {label} ({path})")

    print()
    print(f"{'Model':<22} {'Input(H x W)':<14} {'Params(M)':>10} {'GMACs':>8} {'GFLOPs':>8} "
          f"{'Latency(ms)':>12} {'img/s':>8}")
    print("-" * 90)
    for r in rows:
        print(f"{r['label']:<22} {r['hw']:<14} {r['params_m']:>10.3f} {r['gmacs']:>8.4f} "
              f"{r['gflops']:>8.4f} {r['latency_ms']:>12.3f} {r['fps']:>8.2f}")
    print("-" * 90)
    print(f"device={args.device}, batch={args.batch}, provider={rows[0]['provider'] if rows else '?'} "
          f"(GFLOPs includes each model's own decode ops, same convention as profile_onnx.py)")


if __name__ == "__main__":
    main()
