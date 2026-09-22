"""PicoDet with a PP-LCNet backbone, on top of LibreYOLO's CSP-PAN neck + PicoHead.

PP-LCNet (v1) is re-implemented from PaddleDetection's ppdet/modeling/backbones/lcnet.py.
Module names match Paddle's (conv1.conv, blocks2..6.N.dw_conv/se/pw_conv ...), so the
official Paddle ImageNet weights load with a trivial key rename (_mean -> running_mean,
_variance -> running_var).

Why a subclass of LibrePICODET instead of patching a loaded model:
train() calls _rebuild_for_new_classes() whenever your dataset's class count is not 80,
and that re-runs self._init_model(). The backbone therefore has to be built inside
_init_model(), otherwise it silently turns back into ESNet.

Usage:
    python picodet_lcnet_backbone.py --data my-dataset.yaml --device 0
    python picodet_lcnet_backbone.py --data coco8.yaml --epochs 3 --batch 4 --device cpu
    python picodet_lcnet_backbone.py --data my.yaml --lcnet-weights none      # random init
    python picodet_lcnet_backbone.py --data my.yaml --lcnet-weights /path/PPLCNet_x0_75_pretrained.pdparams

Reload a trained checkpoint with the SAME class and scale (LibreYOLO("best.pt") would
build an ESNet model):
    model = PicoDetLCNet("runs/train/<name>/weights/best.pt", size="s", lcnet_scale=0.75)
"""
import argparse
import inspect
import os
import pickle
import re
import urllib.request

import numpy as np
import torch
import torch.nn as nn
from libreyolo.models.picodet.model import LibrePICODET
from libreyolo.models.picodet.nn import CSPPAN, SIZE_SPEC, LibrePICODETModel
import picodet_letterbox                                              # ADD THIS LINE

from picodet_activation import ACTS, GATES, make_act, make_gate, set_activation  # keep this file next to the script

# ---------------------------------------------------------------------------
# PP-LCNet (v1) backbone, mirrors PaddleDetection's LCNet
# ---------------------------------------------------------------------------

# k, in_c, out_c, stride, use_se
NET_CONFIG = {
    "blocks2": [[3, 16, 32, 1, False]],
    "blocks3": [[3, 32, 64, 2, False], [3, 64, 64, 1, False]],
    "blocks4": [[3, 64, 128, 2, False], [3, 128, 128, 1, False]],
    "blocks5": [[3, 128, 256, 2, False]] + [[5, 256, 256, 1, False]] * 5,
    "blocks6": [[5, 256, 512, 2, True], [5, 512, 512, 1, True]],
}


def make_divisible(v, divisor=8, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class ConvBNLayer(nn.Module):
    def __init__(self, in_ch, out_ch, k, stride, groups=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, stride, padding=(k - 1) // 2,
                              groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = make_act()  # hard-swish unless set_activation() was called

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class SEModule(nn.Module):
    def __init__(self, channel, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(channel, channel // reduction, 1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(channel // reduction, channel, 1)
        self.hardsigmoid = make_gate()  # nn.Hardsigmoid (x/6 + 0.5, as Paddle 2.x) unless changed

    def forward(self, x):
        s = self.hardsigmoid(self.conv2(self.relu(self.conv1(self.avg_pool(x)))))
        return x * s


class DepthwiseSeparable(nn.Module):
    def __init__(self, in_ch, out_ch, stride, dw_size=3, use_se=False):
        super().__init__()
        self.use_se = use_se
        self.dw_conv = ConvBNLayer(in_ch, in_ch, dw_size, stride, groups=in_ch)
        if use_se:
            self.se = SEModule(in_ch)
        self.pw_conv = ConvBNLayer(in_ch, out_ch, 1, 1)

    def forward(self, x):
        x = self.dw_conv(x)
        if self.use_se:
            x = self.se(x)
        return self.pw_conv(x)


class LCNet(nn.Module):
    """Returns the feature maps listed in ``feature_maps`` (default 3,4,5 = strides 8/16/32)."""

    def __init__(self, scale=0.75, feature_maps=(3, 4, 5)):
        super().__init__()
        self.scale = scale
        self.feature_maps = tuple(feature_maps)
        self.conv1 = ConvBNLayer(3, make_divisible(16 * scale), 3, 2)

        out_channels = []
        for name in ("blocks2", "blocks3", "blocks4", "blocks5", "blocks6"):
            cfg = NET_CONFIG[name]
            setattr(self, name, nn.Sequential(*[
                DepthwiseSeparable(make_divisible(i * scale), make_divisible(o * scale),
                                   stride=s, dw_size=k, use_se=se)
                for (k, i, o, s, se) in cfg
            ]))
            out_channels.append(make_divisible(cfg[-1][2] * scale))
        # out_channels[0] is blocks2's output, which Paddle never returns
        self.out_channels = tuple(
            ch for idx, ch in enumerate(out_channels[1:]) if idx + 2 in self.feature_maps
        )

    def forward(self, x):
        outs = []
        x = self.blocks2(self.conv1(x))
        x = self.blocks3(x); outs.append(x)   # stride 4
        x = self.blocks4(x); outs.append(x)   # stride 8
        x = self.blocks5(x); outs.append(x)   # stride 16
        x = self.blocks6(x); outs.append(x)   # stride 32
        return [o for i, o in enumerate(outs) if i + 2 in self.feature_maps]


# ---------------------------------------------------------------------------
# Paddle weight loading
# ---------------------------------------------------------------------------

PRETRAINED = {
    # PicoDet-LCNet configs in PaddleDetection point at these
    0.75: "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/PPLCNet_x0_75_pretrained.pdparams",
    0.35: "https://paddledet.bj.bcebos.com/models/pretrained/LCNet_x0_35_pretrained.pdparams",
}


def _read_pdparams(path):
    # .pdparams is a pickle: only load files from a source you trust.
    try:
        with open(path, "rb") as f:
            obj = pickle.load(f, encoding="latin1")
    except Exception:
        import paddle  # fallback if plain pickle can't read it
        obj = paddle.load(path, return_numpy=True)
    return {k: np.asarray(v) for k, v in obj.items() if isinstance(v, np.ndarray)}


def load_paddle_lcnet(backbone, src):
    """Load a Paddle PP-LCNet ImageNet checkpoint (path or URL) into ``backbone``."""
    if src.startswith(("http://", "https://")):
        cache = os.path.join(os.path.expanduser("~"), ".cache", "picodet_lcnet")
        os.makedirs(cache, exist_ok=True)
        path = os.path.join(cache, os.path.basename(src))
        if not os.path.exists(path):
            print(f"downloading {src}")
            urllib.request.urlretrieve(src, path)
    else:
        path = src

    sd = {}
    for k, v in _read_pdparams(path).items():
        k = k[len("backbone."):] if k.startswith("backbone.") else k
        k = k.replace(".bn._mean", ".bn.running_mean").replace(".bn._variance", ".bn.running_var")
        sd[k] = torch.from_numpy(v.astype(np.float32))

    own = backbone.state_dict()
    matched = {k: v for k, v in sd.items() if k in own and own[k].shape == v.shape}
    missing = [k for k in own if k not in matched and not k.endswith("num_batches_tracked")]
    if missing:
        raise RuntimeError(
            f"{len(missing)} backbone tensors not found in {path} "
            f"(scale mismatch or unexpected key names), e.g. {missing[:5]}"
        )
    backbone.load_state_dict(matched, strict=False)
    print(f"loaded {len(matched)} PP-LCNet tensors from {path}")


# ---------------------------------------------------------------------------
# PicoDet with the LCNet backbone
# ---------------------------------------------------------------------------


def _kaiming_init(module):
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)


def _act_candidates():
    """Activation names to try for the neck.

    Different LibreYOLO versions spell the hard-swish name differently ("hswish",
    "hardswish", ...). Read the name the installed parent class uses for its own neck
    first, then fall back to common spellings.
    """
    names = []
    try:
        m = re.search(r"act\s*=\s*[\"']([^\"']+)[\"']",
                      inspect.getsource(LibrePICODETModel.__init__))
        if m:
            names.append(m.group(1))
    except (OSError, TypeError):
        pass
    for n in ("hswish", "hardswish", "h_swish", "hard_swish", "HSwish", "Hardswish"):
        if n not in names:
            names.append(n)
    return names


def _build_neck(in_channels, out_channels):
    err = None
    for act in _act_candidates():
        try:
            return CSPPAN(
                in_channels=in_channels, out_channels=out_channels,
                kernel_size=5, num_features=4, expansion=1.0,
                num_csp_blocks=1, use_depthwise=True, act=act,
            )
        except ValueError as e:  # "Unsupported activation: ..." -> try the next spelling
            if "activation" not in str(e).lower():
                raise
            err = e
    raise RuntimeError(f"none of {_act_candidates()} is accepted by the installed CSPPAN") from err


class PicoDetLCNetModel(LibrePICODETModel):
    def __init__(self, size="s", nb_classes=80, lcnet_scale=0.75, lcnet_weights=None):
        # Parent builds ESNet + neck + head. We keep its head and replace backbone + neck.
        super().__init__(size=size, nb_classes=nb_classes)
        spec = SIZE_SPEC[size]

        self.backbone = LCNet(scale=lcnet_scale, feature_maps=(3, 4, 5))
        # Neck input widths come from backbone.out_channels, so the neck must be rebuilt.
        # Head input is spec["neck_ch"], unchanged, so the head is kept.
        self.neck = _build_neck(self.backbone.out_channels, spec["neck_ch"])
        _kaiming_init(self.backbone)
        _kaiming_init(self.neck)
        if lcnet_weights:  # after init, so pretrained weights are not overwritten
            load_paddle_lcnet(self.backbone, lcnet_weights)


class PicoDetLCNet(LibrePICODET):
    """LibrePICODET wrapper that builds PicoDetLCNetModel.

    Constructor kwargs become attributes before _init_model() runs:
        PicoDetLCNet(size="s", lcnet_scale=0.75, lcnet_weights="auto" | "none" | path | url)
    """

    lcnet_scale = 0.75
    lcnet_weights = "auto"

    def _init_model(self):
        # Same skip-download rule RT-DETR uses: no pretrained fetch when a checkpoint is
        # about to overwrite the weights, or when rebuilding for a new class count (the
        # old state_dict, backbone included, is copied back afterwards).
        skip = (
            getattr(self, "_in_rebuild", False)
            or getattr(self, "_loading_from_weights", False)
            or getattr(self, "_initializing_from_scratch", False)
        )
        weights = self.lcnet_weights
        if skip or weights in (None, "none", "None", ""):
            weights = None
        elif weights == "auto":
            weights = PRETRAINED.get(self.lcnet_scale)  # None if no known URL for this scale
        return PicoDetLCNetModel(size=self.size, nb_classes=self.nb_classes,
                                 lcnet_scale=self.lcnet_scale, lcnet_weights=weights)

    def _get_available_layers(self):
        # Parent references backbone.blocks, which LCNet does not have.
        b = self.model.backbone
        return {
            "backbone_conv1": b.conv1,
            "backbone_blocks": nn.ModuleList([b.blocks2, b.blocks3, b.blocks4, b.blocks5, b.blocks6]),
            "neck": self.model.neck,
            "head": self.model.head,
        }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--size", default="s", choices=["s", "m", "l"])
    p.add_argument("--lcnet-scale", type=float, default=0.75)
    p.add_argument("--lcnet-weights", default="auto", help="auto | none | path | url")
    p.add_argument("--act", default="hswish", choices=sorted(ACTS),
                   help="activation for backbone + neck + head (pretrained weights assume hswish)")
    p.add_argument("--gate", default="default", choices=sorted(GATES),
                   help="SE-gate replacing the hard sigmoid (LCNet SE blocks; ESNet only if used)")
    p.add_argument("--device", default="")
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr0", type=float, default=0.1)
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args()

    set_activation(args.act, gate=args.gate)  # must run before the model is built
    model = PicoDetLCNet(size=args.size, lcnet_scale=args.lcnet_scale,
                         lcnet_weights=args.lcnet_weights)
    n_bb = sum(x.numel() for x in model.model.backbone.parameters()) / 1e6
    n_all = sum(x.numel() for x in model.model.parameters()) / 1e6
    print(f"backbone params: {n_bb:.2f}M | total: {n_all:.2f}M | input size: {model.input_size}")

    print(model.train(data=args.data, epochs=args.epochs, batch=args.batch, lr0=args.lr0,
                      device=args.device, workers=args.workers, imgsz=(320, 448)))
    print(model.val(data=args.data, imgsz=(320, 448)))


if __name__ == "__main__":  # guard needed for dataloader workers / multi-GPU spawn
    main()
