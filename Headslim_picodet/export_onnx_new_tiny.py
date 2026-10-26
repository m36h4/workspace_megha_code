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
import onnx
import torch

import picodet_letterbox as pl  # noqa: F401  -- must be imported before building the model


def load_model(args):
    from picodet_activation import set_activation
    set_activation(args.act, gate=args.gate)

    if args.arch == "esnet":
        from libreyolo.models.picodet.model import LibrePICODET
        return LibrePICODET(args.weights, size=args.size)
    elif args.arch == "esnet_small":
        from picodet_esnet_small import load_small
        return load_small(args.weights, size=args.size)
    elif args.arch == "esnet_tiny":
        from picodet_esnet_tiny import load_tiny
        return load_tiny(args.weights, size=args.size, preset=args.tiny_preset)
    elif args.arch == "mbv2":
        from picodet_mbv2 import load_mbv2
        return load_mbv2(args.weights, backbone_act=args.backbone_act, size=args.size)
    elif args.arch == "esnet_headslim":
        from picodet_esnet_headslim import PicoDetHeadSlim
        return PicoDetHeadSlim(args.weights, size=args.size)
    else:
        from picodet_lcnet_backbone import PicoDetLCNet
        return PicoDetLCNet(args.weights, size=args.size, lcnet_scale=args.lcnet_scale,
                            act=args.act, gate=args.gate)



class RawLevels(torch.nn.Module):
    """Backbone+neck+head WITHOUT decode. One channels-first tensor per pyramid level:
    cat([cls logits, box-distribution logits], dim=1). Only Conv/act/Concat remain in the
    head -- no Transpose/Reshape/Softmax/Gather/ReduceSum for the compiler to trip on.
    Decode on the host with picodet_raw_decode.decode_levels()."""

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        cls_scores, bbox_preds = self.net(x)          # head.export must be False
        return tuple(torch.cat([c, b], dim=1) for c, b in zip(cls_scores, bbox_preds))


def activation_census(net, requested_act):
    """Prints which activation classes are INSIDE the model about to be exported, so an
    ignored --act can't hide (the .pt file doesn't record its activation -- the code that
    builds the model decides it)."""
    from collections import Counter
    from picodet_activation import ACTS
    hints = ("swish", "silu", "relu", "sigmoid", "mish", "gelu", "hardswish", "hsigmoid", "hswish")
    c = Counter()
    for m in net.modules():
        if any(True for _ in m.children()):
            continue
        t = type(m)
        if t.__module__.startswith("torch.nn.modules.activation") or any(h in t.__name__.lower() for h in hints):
            c[t.__name__] += 1
    print("Activation census of the model being exported: "
          + ", ".join(f"{k} x{v}" for k, v in sorted(c.items())))
    wanted = type(ACTS[requested_act]()).__name__
    if requested_act != "relu" and c.get(wanted, 0) == 0:
        print(f"*** WARNING: --act {requested_act} asked for {wanted}, but the model contains NONE. "
              f"--act was NOT honored here (or an old copy of the model file is being imported). ***")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True)
    p.add_argument("--arch", required=True, choices=["esnet", "esnet_headslim", "esnet_small", "esnet_tiny", "lcnet", "mbv2"])
    p.add_argument("--backbone-act", default="relu6", choices=["relu6", "relu"],
                   help="mbv2 only: must match what the model was trained with")
    p.add_argument("--tiny-preset", default="t1", choices=["t0", "t1", "t2", "t3"],
                   help="esnet_tiny only: must match training")
    p.add_argument("--size", default="s", choices=["s", "m", "l"])
    p.add_argument("--lcnet-scale", type=float, default=0.75)
    p.add_argument("--act", default="leakyrelu")
    p.add_argument("--gate", default="sigmoid")
    p.add_argument("--imgsz", type=int, nargs=2, default=[320, 480], metavar=("H", "W"))
    p.add_argument("--out", default=None, help="output .onnx path (default: derived from --weights)")
    p.add_argument("--opset", type=int, default=17)
    p.add_argument("--head-style", default="stock", choices=["stock", "raw"],
                   help="stock: decode inside the graph, one (1,N,4+nc) output. "
                        "raw: no decode in the graph, one (1,nc+4*(reg_max+1),h,w) output per stride; "
                        "decode on the host (picodet_raw_decode.py, picodet_eval_onnx.py does it automatically)")
    p.add_argument("--dynamic-batch", action="store_true", help="allow batch size != 1 at inference")
    args = p.parse_args()

    out_path = args.out or args.weights.rsplit(".", 1)[0] + f"_{args.imgsz[0]}x{args.imgsz[1]}.onnx"

    model = load_model(args)
    net = model.model.eval().to("cpu")  # export from CPU: GPU-loaded checkpoints (your case) would
                                        # otherwise crash with a device mismatch against the dummy
                                        # input, and CPU tracing avoids CUDA-specific op quirks too.
    activation_census(net, args.act)
    h, w = args.imgsz
    dummy = torch.randn(1, 3, h, w)  # CPU, matching net above
    raw = args.head_style == "raw"

    # reference (stock, decoded in torch) -- used to verify BOTH styles
    net.head.export = True
    with torch.no_grad():
        torch_out = net(dummy)
    print(f"PyTorch forward at {h}x{w}: stock output shape {tuple(torch_out.shape)}")

    strides = tuple(net.head.strides)
    nc, reg_max = net.head.num_classes, net.head.reg_max
    if raw:
        net.head.export = False
        export_net = RawLevels(net).eval()
        output_names = [f"out_s{st}" for st in strides]
        dynamic_axes = {n: {0: "batch"} for n in ["images"] + output_names} if args.dynamic_batch else None
    else:
        export_net = net
        output_names = ["output"]
        dynamic_axes = {"images": {0: "batch"}, "output": {0: "batch"}} if args.dynamic_batch else None
    torch.onnx.export(
        export_net, dummy, out_path,
        input_names=["images"], output_names=output_names,
        opset_version=args.opset, do_constant_folding=True,
        dynamic_axes=dynamic_axes,
    )

    # PyTorch's exporter can silently write weights to a separate "<out_path>.data" file
    # instead of inlining them, even for a model this small. If that companion file isn't
    # shared alongside the .onnx, loading it elsewhere fails with a confusing
    # "tensor ... should be stored in X.onnx.data, but it doesn't exist" error -- exactly
    # what happened when only the .onnx was sent to a client. Fold everything back into
    # one self-contained file so there's nothing extra to remember to send.
    import os
    data_path = out_path + ".data"
    if os.path.exists(data_path):
        reloaded = onnx.load(out_path)  # reads both pieces back in
        onnx.save_model(reloaded, out_path, save_as_external_data=False)
        os.remove(data_path)
        print(f"exported: {out_path}  (folded external weights back into one file)")
    else:
        print(f"exported: {out_path}")

    # Verify before handing it back -- an export that "succeeds" but silently diverges
    # numerically from the source model is worse than one that visibly fails.
    import onnxruntime as ort
    sess = ort.InferenceSession(out_path, providers=["CPUExecutionProvider"])
    onnx_outs = sess.run(None, {sess.get_inputs()[0].name: dummy.numpy()})
    if raw:
        from picodet_raw_decode import decode_levels
        for n, o in zip(output_names, onnx_outs):
            print(f"  {n}: {o.shape}")
        onnx_out = decode_levels(onnx_outs, strides, nc, reg_max)[None]
        print("(host decode of the raw outputs compared against the stock in-graph decode)")
    else:
        onnx_out = onnx_outs[0]
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
