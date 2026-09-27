"""Breaks down a .pt PicoDet checkpoint's MACs by MODULE TYPE (Conv2d, BatchNorm2d,
LeakyReLU, Sigmoid, etc.) -- same table format as diagnose_onnx.py, so you can put
the two side by side and see exactly which op category CHANGED when exporting to
ONNX (the decode math, for one, only exists in the ONNX version and shows up there
as new Softmax/Sub/Add/Gather rows with no .pt counterpart).

Reuses profile_picodet.py's own load_model()/detect_backbone_kind() -- same
--headslim-script / --lcnet-script flags, same checkpoint auto-detection.

Usage:
    python diagnose_picodet.py --weights runs/train/picodet_exp/weights/best.pt \\
        --headslim-script picodet_esnet_headslim.py --imgsz 320x480 --act leakyrelu --gate sigmoid
"""
import argparse
from collections import defaultdict

from torchinfo import summary as torchinfo_summary

from profile_picodet import detect_backbone_kind, import_picodet_headslim, import_picodet_lcnet, parse_imgsz


def load_model(weights_path, device, size_override=None, lcnet_script=None, lcnet_scale=0.75,
               headslim_script=None):
    import torch
    ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
    kind = detect_backbone_kind(state_dict)

    size = size_override or (ckpt.get("size") if isinstance(ckpt, dict) else None)
    nb_classes = ckpt.get("nc") if isinstance(ckpt, dict) else None
    kwargs = {}
    if size is not None:
        kwargs["size"] = size
    if nb_classes is not None:
        kwargs["nb_classes"] = nb_classes

    if kind == "lcnet":
        PicoDetLCNet = import_picodet_lcnet(lcnet_script)
        kwargs["lcnet_scale"] = lcnet_scale
        kwargs.setdefault("size", "s")
        return PicoDetLCNet(model_path=weights_path, device=device, **kwargs)

    from profile_picodet import detect_head_stacked_convs
    if detect_head_stacked_convs(state_dict) == 1:
        if headslim_script is None:
            raise SystemExit("Slimmed head detected -- pass --headslim-script.")
        PicoDetHeadSlim = import_picodet_headslim(headslim_script)
        return PicoDetHeadSlim(model_path=weights_path, device=device, **kwargs)

    from libreyolo.models.picodet.model import LibrePICODET
    return LibrePICODET(model_path=weights_path, device=device, **kwargs)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True)
    p.add_argument("--imgsz", type=str, default=None, help="e.g. 320x480 (WxH, same as profile_picodet.py)")
    p.add_argument("--size", default=None)
    p.add_argument("--act", default="hswish")
    p.add_argument("--gate", default="default")
    p.add_argument("--lcnet-script", default=None)
    p.add_argument("--lcnet-scale", type=float, default=0.75)
    p.add_argument("--headslim-script", default=None)
    args = p.parse_args()

    from picodet_activation import set_activation
    set_activation(args.act, gate=args.gate)

    model = load_model(args.weights, device="cpu", size_override=args.size,
                       lcnet_script=args.lcnet_script, lcnet_scale=args.lcnet_scale,
                       headslim_script=args.headslim_script)

    imgsz_hw = parse_imgsz(args.imgsz) or model.input_size
    if not isinstance(imgsz_hw, (list, tuple)):
        imgsz_hw = (imgsz_hw, imgsz_hw)
    h, w = imgsz_hw

    s = torchinfo_summary(model.model.eval(), input_size=(1, 3, h, w), verbose=0)

    by_type = defaultdict(lambda: [0, 0])
    for row in s.summary_list:
        if not row.is_leaf_layer:
            continue
        by_type[row.class_name][0] += 1
        by_type[row.class_name][1] += row.macs

    total = sum(v[1] for v in by_type.values())
    print(f"{'Module type':<20} {'Count':>6} {'Total MACs':>14} {'% of total':>10}")
    print("-" * 55)
    for cls, (count, macs) in sorted(by_type.items(), key=lambda x: -x[1][1]):
        pct = 100 * macs / total if total else 0
        print(f"{cls:<20} {count:>6} {macs:>14,.0f} {pct:>9.2f}%")
    print("-" * 55)
    print(f"{'TOTAL':<20} {sum(v[0] for v in by_type.values()):>6} {total:>14,.0f}")
    print(f"\nGMACs: {total/1e9:.4f}  |  leaf-layer count: {sum(v[0] for v in by_type.values())}  |"
          f"  params: {sum(p.numel() for p in model.model.parameters()):,}")


if __name__ == "__main__":
    main()
