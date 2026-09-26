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
    p.add_argument("--imgsz", type=str, default=None)
    p.add_argument("--device", default="auto", help="cpu | cuda | cuda:0 | auto")
    p.add_argument("--size", default=None, choices=["xs", "s", "m", "l"])
    p.add_argument("--lcnet-script", default=None)
    p.add_argument("--lcnet-scale", type=float, default=0.75)
    p.add_argument("--headslim-script", default=None,
                   help="Path to picodet_esnet_headslim.py. Needed to correctly load a "
                        "checkpoint whose head was trained with stacked_convs=1 -- same "
                        "role as --lcnet-script, one level down (head, not backbone).")
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--runs", type=int, default=100)
    p.add_argument("--half", action="store_true")
    p.add_argument("--backend", default="torchinfo", choices=["torchinfo", "thop"])
    return p.parse_args()


def parse_imgsz(imgsz_arg):
    if imgsz_arg is None:
        return None
    s = str(imgsz_arg).lower().replace(",", "x")
    if "x" in s:
        w_str, h_str = s.split("x")
        w, h = int(w_str), int(h_str)
        return (h, w)
    v = int(s)
    return (v, v)


def detect_head_stacked_convs(state_dict):
    """Checks whether the head has one or two stacked convs per level by looking for
    the SECOND stacked conv's keys (head.cls_convs.0.1.*). Present -> stock (2).
    Absent -> slimmed (1), e.g. a checkpoint trained with picodet_esnet_headslim.py.
    Only meaningful for the ESNet family; PicoDetLCNet checkpoints aren't affected by
    this specific lever."""
    return 2 if any(k.startswith("head.cls_convs.0.1.") for k in state_dict) else 1


def import_picodet_headslim(script_path):
    """Dynamically imports PicoDetHeadSlim from a user-supplied script path, same
    pattern as import_picodet_lcnet() below."""
    script_path = Path(script_path).resolve()
    if not script_path.is_file():
        raise SystemExit(f"--headslim-script not found: {script_path}")
    script_dir = str(script_path.parent)
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
    spec = importlib.util.spec_from_file_location("_picodet_headslim_module", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "PicoDetHeadSlim"):
        raise SystemExit(
            f"{script_path} was imported but has no PicoDetHeadSlim class. "
            "Pass the correct --headslim-script path."
        )
    return module.PicoDetHeadSlim


def detect_backbone_kind(state_dict):
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
        raise SystemExit(f"{script_path} was imported but has no PicoDetLCNet class.")
    return module.PicoDetLCNet


def load_model(weights_path, device, size_override=None, lcnet_script=None, lcnet_scale=0.75,
               headslim_script=None):
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
        print("Detected PP-LCNet-style backbone keys.")
        if lcnet_script is None:
            raise SystemExit("Pass --lcnet-script for an LCNet checkpoint.")
        PicoDetLCNet = import_picodet_lcnet(lcnet_script)
        kwargs["lcnet_scale"] = lcnet_scale
        kwargs.setdefault("size", "s")
        model = PicoDetLCNet(model_path=weights_path, device=device, **kwargs)
    elif kind == "esnet":
        stacked_convs = detect_head_stacked_convs(state_dict)
        if stacked_convs == 1:
            print("Detected a SLIMMED head (missing head.cls_convs.<lvl>.1.* keys, "
                  "stacked_convs=1) -- this needs PicoDetHeadSlim, not stock LibrePICODET.")
            if headslim_script is None:
                raise SystemExit(
                    "This checkpoint was trained with a slimmed head "
                    "(picodet_esnet_headslim.py's PicoDetHeadSlim). Re-run with:\n"
                    "  --headslim-script /path/to/picodet_esnet_headslim.py"
                )
            PicoDetHeadSlim = import_picodet_headslim(headslim_script)
            model = PicoDetHeadSlim(model_path=weights_path, device=device, **kwargs)
        else:
            from libreyolo.models.picodet.model import LibrePICODET
            model = LibrePICODET(model_path=weights_path, device=device, **kwargs)
    else:
        raise SystemExit("Could not determine backbone kind from checkpoint key names.")

    model.model.eval()
    return model


def count_params(nn_module):
    total = sum(p.numel() for p in nn_module.parameters())
    trainable = sum(p.numel() for p in nn_module.parameters() if p.requires_grad)
    return total, trainable


def count_macs_torchinfo(nn_module, imgsz_hw, device):
    from torchinfo import summary as torchinfo_summary
    h, w = imgsz_hw
    nn_module_eval = nn_module.eval()
    s = torchinfo_summary(
        nn_module_eval, input_size=(1, 3, h, w), device=device,
        col_names=("input_size", "output_size", "num_params", "mult_adds"),
        depth=4, verbose=0,
    )
    print(s)
    return s.total_mult_adds, s.total_params


def count_macs_thop(nn_module, imgsz_hw, device):
    from thop import profile as thop_profile
    h, w = imgsz_hw
    dummy = torch.randn(1, 3, h, w, device=device)
    nn_module_eval = nn_module.eval()
    with torch.no_grad():
        macs, params = thop_profile(nn_module_eval, inputs=(dummy,), verbose=False)
    for m in nn_module_eval.modules():
        if hasattr(m, "total_ops"):
            del m.total_ops
        if hasattr(m, "total_params"):
            del m.total_params
    return macs, params


def count_macs(nn_module, imgsz_hw, device, backend="torchinfo"):
    if backend == "thop":
        return count_macs_thop(nn_module, imgsz_hw, device)
    return count_macs_torchinfo(nn_module, imgsz_hw, device)


def time_inference(model, imgsz_hw, device, batch, warmup, runs, half):
    import numpy as np
    from PIL import Image
    h, w = imgsz_hw
    dummy_img = Image.fromarray((np.random.rand(h, w, 3) * 255).astype("uint8"), mode="RGB")
    if half and device.type == "cuda":
        model.model.half()

    with torch.no_grad():
        for _ in range(warmup):
            tensors = []
            for _ in range(batch):
                t, _o, _s, _r = model._preprocess(dummy_img, "auto", input_size=imgsz_hw)
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
                t, _o, _s, _r = model._preprocess(dummy_img, "auto", input_size=imgsz_hw)
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
        return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    return torch.device(device_arg)


def main():
    args = parse_args()
    device = resolve_device(args.device)
    print(f"Loading checkpoint: {args.weights}")
    model = load_model(args.weights, device=str(device), size_override=args.size,
                       lcnet_script=args.lcnet_script, lcnet_scale=args.lcnet_scale,
                       headslim_script=args.headslim_script)

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

    total_params, trainable_params = count_params(model.model)
    print(f"Total params      : {total_params:,}  ({total_params/1e6:.3f} M)")
    print(f"Trainable params  : {trainable_params:,}  ({trainable_params/1e6:.3f} M)")

    print("-" * 60)
    print(f"Params/MACs summary (backend={args.backend})")
    macs, backend_params = count_macs(model.model, imgsz_hw, device, backend=args.backend)
    gmacs = macs / 1e9
    gflops = gmacs * 2
    print("-" * 60)
    print(f"GMACs             : {gmacs:.4f}")
    print(f"GFLOPs (2x GMACs) : {gflops:.4f}")
    print(f"({args.backend} param check: {backend_params:,.0f})")

    print("-" * 60)
    print(f"Timing: warmup={args.warmup}, runs={args.runs}, batch={args.batch}, half={args.half}")
    latencies_ms = time_inference(model, imgsz_hw, device, args.batch, args.warmup, args.runs, args.half)
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
