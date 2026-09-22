#!/usr/bin/env python3
"""
profile_picodet.py
-------------------
Reports parameter count, GMACs/GFLOPs, and inference latency/throughput for a
LibreYOLO PicoDet checkpoint (e.g. last.pt / best.pt). Auto-detects whether
the checkpoint uses LibreYOLO's stock ESNet backbone or a custom PP-LCNet
backbone (e.g. ch_act_picodet_lcnet_backbone.py's PicoDetLCNet) and loads it
with the right class.

Supports two backends for the params/MACs summary, selected with --backend:
  torchinfo (default) -- pip install torchinfo
  thop                -- pip install thop

Usage:
    python profile_picodet.py --weights last.pt
    python profile_picodet.py --weights last.pt --imgsz 320 --device cuda --runs 200
    python profile_picodet.py --weights last.pt --backend thop
"""

import argparse
import importlib.util
import re
import sys
import time
import statistics as stats
from pathlib import Path

import torch


def parse_args():
    p = argparse.ArgumentParser(description="Profile a LibreYOLO PicoDet .pt checkpoint")
    p.add_argument("--weights", required=True, help="Path to .pt checkpoint (e.g. last.pt)")
    p.add_argument("--imgsz", type=str, default=None,
                   help="Input size. Either a single int for square (e.g. 320) or "
                        "WxH for rectangular (e.g. 320x480 meaning width=320, height=480). "
                        "Defaults to the size stored in the checkpoint.")
    p.add_argument("--device", default="auto", help="cpu | cuda | cuda:0 | auto")
    p.add_argument("--size", default=None, choices=["xs", "s", "m", "l"],
                   help="Override PicoDet size (xs/s/m/l). Default: auto-detect from checkpoint.")
    p.add_argument("--lcnet-script", default=None,
                   help="Path to ch_act_picodet_lcnet_backbone.py (or wherever your "
                        "PicoDetLCNet subclass lives). Required only for checkpoints "
                        "trained with the PP-LCNet backbone script -- auto-detected from "
                        "the checkpoint's key names, but the script location can't be "
                        "guessed, so pass it explicitly the first time you profile such "
                        "a checkpoint. The directory containing this file must also "
                        "contain picodet_activation.py (its sibling import).")
    p.add_argument("--lcnet-scale", type=float, default=0.75,
                   help="lcnet_scale the checkpoint was trained with (PicoDetLCNet only).")
    p.add_argument("--batch", type=int, default=1, help="Batch size for timing")
    p.add_argument("--warmup", type=int, default=20, help="Warmup iterations before timing")
    p.add_argument("--runs", type=int, default=100, help="Timed iterations")
    p.add_argument("--half", action="store_true", help="Run timing in fp16 (CUDA only)")
    p.add_argument("--backend", default="torchinfo", choices=["torchinfo", "thop"],
                   help="Library used for the params/MACs summary (default: torchinfo).")
    return p.parse_args()


def parse_imgsz(imgsz_arg):
    """Parses --imgsz into a (h, w) tuple, the order LibreYOLO's picodet
    preprocessing expects. Accepts a single int ('320' -> square (320,320))
    or 'WxH' / 'W,H' (e.g. '320x480' meaning width=320, height=480 ->
    returned as (480, 320))."""
    if imgsz_arg is None:
        return None
    s = str(imgsz_arg).lower().replace(",", "x")
    if "x" in s:
        w_str, h_str = s.split("x")
        w, h = int(w_str), int(h_str)
        return (h, w)  # LibreYOLO's preprocess_numpy unpacks as (input_h, input_w)
    v = int(s)
    return (v, v)


def detect_backbone_kind(state_dict):
    """Inspects checkpoint key names to tell LibreYOLO's stock ESNet backbone
    apart from a custom PP-LCNet backbone (e.g. the PicoDetLCNet subclass in
    ch_act_picodet_lcnet_backbone.py).

    ESNet   (LibrePICODET stock): backbone.blocks.<i>.conv_dw_1 / conv_pw_2 / ...
    PP-LCNet (PicoDetLCNet):      backbone.blocks2.<i>.dw_conv / pw_conv / ...
                                   (stage-named blocks2..blocks6, each block has
                                   only dw_conv[+se]+pw_conv -- no split/concat
                                   conv_dw_1/conv_pw_2/conv_linear_1/2 pattern)

    Returns "esnet", "lcnet", or "unknown".
    """
    keys = list(state_dict.keys())
    has_esnet_pattern = any(".conv_dw_1." in k or ".conv_pw_2." in k for k in keys)
    has_lcnet_pattern = any(
        re.match(r"^backbone\.blocks[2-6]\.\d+\.(dw_conv|pw_conv)\.", k) for k in keys
    )
    if has_esnet_pattern and not has_lcnet_pattern:
        return "esnet"
    if has_lcnet_pattern and not has_esnet_pattern:
        return "lcnet"
    return "unknown"


def import_picodet_lcnet(script_path):
    """Dynamically imports PicoDetLCNet from a user-supplied script path.
    The script's sibling import (picodet_activation.py) must sit in the same
    directory -- this adds that directory to sys.path so the import resolves
    the same way it would if you ran the script directly."""
    script_path = Path(script_path).resolve()
    if not script_path.is_file():
        raise SystemExit(f"--lcnet-script not found: {script_path}")

    script_dir = str(script_path.parent)
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)

    spec = importlib.util.spec_from_file_location("_picodet_lcnet_backbone_module", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, "PicoDetLCNet"):
        raise SystemExit(
            f"{script_path} was imported but has no PicoDetLCNet class. "
            "Pass the correct --lcnet-script path."
        )
    return module.PicoDetLCNet


def load_model(weights_path, device, size_override=None, lcnet_script=None, lcnet_scale=0.75):
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
        print("Detected PP-LCNet-style backbone keys (blocksN.i.dw_conv/pw_conv) -- "
              "this checkpoint needs the PicoDetLCNet subclass, not stock LibrePICODET.")
        if lcnet_script is None:
            raise SystemExit(
                "This checkpoint was trained with a custom PP-LCNet backbone "
                "(e.g. ch_act_picodet_lcnet_backbone.py's PicoDetLCNet). Re-run with:\n"
                "  --lcnet-script /path/to/ch_act_picodet_lcnet_backbone.py\n"
                "(and --lcnet-scale if it wasn't the default 0.75)."
            )
        PicoDetLCNet = import_picodet_lcnet(lcnet_script)
        kwargs["lcnet_scale"] = lcnet_scale
        kwargs.setdefault("size", "s")  # PicoDetLCNet requires size; fall back to "s" if unknown
        print(f"Using PicoDetLCNet(size='{kwargs.get('size')}', lcnet_scale={lcnet_scale}, "
              f"nc={nb_classes})")
        model = PicoDetLCNet(model_path=weights_path, device=device, **kwargs)

    elif kind == "esnet":
        from libreyolo.models.picodet.model import LibrePICODET
        if size is not None:
            print(f"Using PicoDet size='{size}'"
                  + (f", nc={nb_classes}" if nb_classes is not None else "")
                  + (" (from --size override)" if size_override else " (from checkpoint metadata)"))
        else:
            print("Could not read 'size' from checkpoint metadata; "
                  "falling back to LibrePICODET's default ('s'). "
                  "If loading fails with a shape mismatch, pass --size explicitly.")
        model = LibrePICODET(model_path=weights_path, device=device, **kwargs)

    else:
        raise SystemExit(
            "Could not determine whether this checkpoint uses LibreYOLO's stock ESNet "
            "backbone or a custom PP-LCNet backbone from its key names. "
            "If this is a PicoDetLCNet checkpoint, pass --lcnet-script explicitly; "
            "otherwise this checkpoint may use yet another custom architecture that "
            "these scripts don't recognize."
        )

    model.model.eval()  # defensive no-op; both classes load with eval() already set
    return model


def count_params(nn_module):
    total = sum(p.numel() for p in nn_module.parameters())
    trainable = sum(p.numel() for p in nn_module.parameters() if p.requires_grad)
    return total, trainable


def count_macs_torchinfo(nn_module, imgsz_hw, device):
    """Uses `torchinfo` to get a full param/MACs summary for one forward pass
    at batch size 1. torchinfo reports `total_mult_adds` (multiply-accumulate
    ops, i.e. MACs). GMACs = total_mult_adds / 1e9; GFLOPs = 2x GMACs (one
    multiply + one add per MAC) -- both are printed. Also prints the
    per-layer table.

    imgsz_hw: (height, width) tuple."""
    try:
        from torchinfo import summary as torchinfo_summary
    except ImportError as e:
        raise SystemExit(
            "The `torchinfo` package is required for this summary.\n"
            "Install it with: pip install torchinfo"
        ) from e

    h, w = imgsz_hw
    nn_module_eval = nn_module.eval()
    stats = torchinfo_summary(
        nn_module_eval,
        input_size=(1, 3, h, w),
        device=device,
        col_names=("input_size", "output_size", "num_params", "mult_adds"),
        depth=4,
        verbose=0,
    )
    print(stats)
    total_params = stats.total_params
    macs = stats.total_mult_adds
    return macs, total_params


def count_macs_thop(nn_module, imgsz_hw, device):
    """Uses `thop` to count MACs (multiply-accumulates) and params for one
    forward pass at batch size 1. GMACs = macs / 1e9; GFLOPs = 2x GMACs (one
    multiply + one add per MAC).

    imgsz_hw: (height, width) tuple."""
    try:
        from thop import profile as thop_profile
    except ImportError as e:
        raise SystemExit(
            "The `thop` package is required for this summary.\n"
            "Install it with: pip install thop"
        ) from e

    h, w = imgsz_hw
    dummy = torch.randn(1, 3, h, w, device=device)
    nn_module_eval = nn_module.eval()
    with torch.no_grad():
        macs, params = thop_profile(nn_module_eval, inputs=(dummy,), verbose=False)

    # thop.profile() attaches total_ops/total_params buffers to every submodule
    # via forward hooks; strip them so they don't linger for later timing runs.
    for m in nn_module_eval.modules():
        if hasattr(m, "total_ops"):
            del m.total_ops
        if hasattr(m, "total_params"):
            del m.total_params

    return macs, params


def count_macs(nn_module, imgsz_hw, device, backend="torchinfo"):
    """Dispatches to the requested backend. Both return (macs, total_params)
    with macs as multiply-accumulate ops (not FLOPs)."""
    if backend == "thop":
        return count_macs_thop(nn_module, imgsz_hw, device)
    return count_macs_torchinfo(nn_module, imgsz_hw, device)


def time_inference(model, imgsz_hw, device, batch, warmup, runs, half):
    """Times the model's own _preprocess -> _forward -> _postprocess stages
    using a synthetic RGB image, mirroring LibreYOLO's own inference path.

    imgsz_hw: (height, width) tuple, forwarded as-is to model._preprocess's
    input_size, which LibreYOLO's picodet preprocessing unpacks as (h, w)."""
    import numpy as np
    from PIL import Image

    h, w = imgsz_hw
    # PIL Image.fromarray expects an array shaped (rows=h, cols=w, channels)
    dummy_img = Image.fromarray(
        (np.random.rand(h, w, 3) * 255).astype("uint8"), mode="RGB"
    )

    if half and device.type == "cuda":
        model.model.half()

    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            tensors = []
            for _ in range(batch):
                t, _orig, _size, _ratio = model._preprocess(dummy_img, "auto", input_size=imgsz_hw)
                tensors.append(t)
            stacked = torch.cat(tensors, dim=0) if batch > 1 else tensors[0]
            stacked = stacked.to(device)
            if half and device.type == "cuda":
                stacked = stacked.half()
            out = model._forward(stacked)
            if device.type == "cuda":
                torch.cuda.synchronize()

    latencies_ms = []
    with torch.no_grad():
        for _ in range(runs):
            tensors = []
            for _ in range(batch):
                t, _orig, _size, _ratio = model._preprocess(dummy_img, "auto", input_size=imgsz_hw)
                tensors.append(t)
            stacked = torch.cat(tensors, dim=0) if batch > 1 else tensors[0]
            stacked = stacked.to(device)
            if half and device.type == "cuda":
                stacked = stacked.half()

            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = model._forward(stacked)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            latencies_ms.append((t1 - t0) * 1000.0)

    return latencies_ms


def resolve_device(device_arg):
    if device_arg == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(device_arg)


def main():
    args = parse_args()
    device = resolve_device(args.device)

    print(f"Loading checkpoint: {args.weights}")
    model = load_model(
        args.weights, device=str(device), size_override=args.size,
        lcnet_script=args.lcnet_script, lcnet_scale=args.lcnet_scale,
    )

    imgsz_hw = parse_imgsz(args.imgsz)
    if imgsz_hw is None:
        native = getattr(model, "input_size", None) or model._get_input_size()
        if isinstance(native, (list, tuple)):
            imgsz_hw = (int(native[0]), int(native[1]))
        else:
            imgsz_hw = (int(native), int(native))
    h, w = imgsz_hw
    print(f"Model family : {getattr(model, 'FAMILY', 'picodet')}  size={getattr(model, 'size', '?')}")
    print(f"Input size   : {w}x{h}  (WxH)  ->  internal (h,w)=({h},{w})")
    print(f"Classes (nc) : {getattr(model, 'nb_classes', '?')}")
    print(f"Device       : {device}")
    print("-" * 60)

    # ---- Params ----
    total_params, trainable_params = count_params(model.model)
    print(f"Total params      : {total_params:,}  ({total_params/1e6:.3f} M)")
    print(f"Trainable params  : {trainable_params:,}  ({trainable_params/1e6:.3f} M)")

    # ---- GMACs / GFLOPs ----
    print("-" * 60)
    print(f"Params/MACs summary (backend={args.backend})")
    macs, backend_params = count_macs(model.model, imgsz_hw, device, backend=args.backend)
    gmacs = macs / 1e9
    gflops = gmacs * 2  # 1 MAC = 2 FLOPs (multiply + add)
    print("-" * 60)
    print(f"GMACs             : {gmacs:.4f}")
    print(f"GFLOPs (2x GMACs) : {gflops:.4f}")
    print(f"({args.backend} param check: {backend_params:,.0f})")

    # ---- Timing ----
    print("-" * 60)
    print(f"Timing: warmup={args.warmup}, runs={args.runs}, batch={args.batch}, half={args.half}")
    latencies_ms = time_inference(
        model, imgsz_hw, device, args.batch, args.warmup, args.runs, args.half
    )
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
    print(f"  Params  : {total_params/1e6:.3f} M")
    print(f"  GMACs   : {gmacs:.4f}")
    print(f"  GFLOPs  : {gflops:.4f}")
    print(f"  Latency : {mean_ms:.3f} ms  ({fps:.2f} img/s)")


if __name__ == "__main__":
    main()
