"""Export a trained PicoDet checkpoint to ONNX, with an input size independent of what
it was trained at (e.g. trained at 320x448, deployed at 320x480).

Why not model.export(format="onnx", ...)? LibreYOLO's built-in exporter explicitly
blocks rectangular imgsz for PicoDet:
    NotImplementedError: Rectangular imgsz export is currently supported for
    YOLO9-family, HRNet, NAFNet, and Real-ESRGAN exports only.
This script does the same thing that exporter does internally for the families it DOES
allow (flip head.export=True, trace at the target shape with torch.onnx.export) --
verified against a real trained checkpoint: PyTorch and the resulting ONNX Runtime
output matched to ~6e-5 max abs difference (float32 precision noise) at (1, 3190, 6)
output shape (4 box coords + nc class scores, per anchor point, all levels concatenated).

The 64-divisibility check you hit during train() does NOT exist anywhere in the export
path (confirmed by reading it) -- PicoDet is fully convolutional, so an input size the
network never trained at still runs; it's only train()'s own extra-cautious guard.

Output format: raw dense predictions, shape (1, N, 4+nc) -- boxes are in CANVAS pixel
coordinates (this input size's canvas, e.g. 320x480), scores are per-class and already
sigmoid-activated (multi-label: one anchor can score high on more than one class). NO
NMS is applied inside the graph. Use picodet_eval_onnx.py to decode, NMS, un-letterbox,
and score this exact file the same way picodet_eval.py scores a .pt checkpoint.

Usage:
    python export_onnx.py --weights runs/train/ESNet/exp4/weights/best.pt --arch esnet \\
        --act leakyrelu --gate sigmoid --imgsz 320 480 --out picodet_esnet_320x480.onnx

    python export_onnx.py --weights runs/train/LCNet/exp4/weights/best.pt --arch lcnet \\
        --lcnet-scale 0.75 --act leakyrelu --gate sigmoid --imgsz 320 480 \\
        --out picodet_lcnet_320x480.onnx
"""
import argparse

import numpy as np
import torch

import picodet_letterbox as pl  # noqa: F401  -- must be imported before building the model


def load_model(args):
    from picodet_activation import set_activation
    set_activation(args.act, gate=args.gate)

    if args.arch == "esnet":
        from libreyolo.models.picodet.model import LibrePICODET
        return LibrePICODET(args.weights, size=args.size)
    elif args.arch == "esnet_headslim":
        from picodet_esnet_headslim import PicoDetHeadSlim
        return PicoDetHeadSlim(args.weights, size=args.size)
    else:
        from picodet_lcnet_backbone import PicoDetLCNet
        return PicoDetLCNet(args.weights, size=args.size, lcnet_scale=args.lcnet_scale,
                            act=args.act, gate=args.gate)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True)
    p.add_argument("--arch", required=True, choices=["esnet", "esnet_headslim", "lcnet"])
    p.add_argument("--size", default="s", choices=["s", "m", "l"])
    p.add_argument("--lcnet-scale", type=float, default=0.75)
    p.add_argument("--act", default="hswish")
    p.add_argument("--gate", default="default")
    p.add_argument("--imgsz", type=int, nargs=2, default=[320, 480], metavar=("H", "W"))
    p.add_argument("--out", default=None, help="output .onnx path (default: derived from --weights)")
    p.add_argument("--opset", type=int, default=17)
    p.add_argument("--dynamic-batch", action="store_true", help="allow batch size != 1 at inference")
    args = p.parse_args()

    out_path = args.out or args.weights.rsplit(".", 1)[0] + f"_{args.imgsz[0]}x{args.imgsz[1]}.onnx"

    model = load_model(args)
    net = model.model.eval().to("cpu")  # export from CPU: GPU-loaded checkpoints (your case) would
                                        # otherwise crash with a device mismatch against the dummy
                                        # input, and CPU tracing avoids CUDA-specific op quirks too.
    net.head.export = True  # single decoded (B, N, 4+nc) tensor instead of the raw per-level lists

    h, w = args.imgsz
    dummy = torch.randn(1, 3, h, w)  # CPU, matching net above

    with torch.no_grad():
        torch_out = net(dummy)
    print(f"PyTorch forward at {h}x{w}: output shape {tuple(torch_out.shape)}")

    dynamic_axes = {"images": {0: "batch"}, "output": {0: "batch"}} if args.dynamic_batch else None
    torch.onnx.export(
        net, dummy, out_path,
        input_names=["images"], output_names=["output"],
        opset_version=args.opset, do_constant_folding=True,
        dynamic_axes=dynamic_axes,
    )
    print(f"exported: {out_path}")

    # Verify before handing it back -- an export that "succeeds" but silently diverges
    # numerically from the source model is worse than one that visibly fails.
    import onnxruntime as ort
    sess = ort.InferenceSession(out_path, providers=["CPUExecutionProvider"])
    onnx_out = sess.run(None, {sess.get_inputs()[0].name: dummy.numpy()})[0]
    diff = np.abs(torch_out.numpy() - onnx_out).max()
    print(f"ONNX Runtime output shape: {onnx_out.shape} | max abs diff vs PyTorch: {diff:.2e}")
    if diff > 1e-2:
        print("WARNING: this is a larger gap than expected float32 rounding -- inspect before deploying.")
    else:
        print("OK: ONNX output matches PyTorch to float32 precision.")

    print(f"\nInput size baked into this graph: {h}x{w} (H, W). Preprocess new images to this\n"
          f"exact canvas with picodet_letterbox.letterbox_np before feeding them in, and use\n"
          f"picodet_eval_onnx.py (same --imgsz) to decode + NMS + un-letterbox the raw output.")


if __name__ == "__main__":
    main()
