"""PicoDet-ESNet with a slimmer head: stacked_convs 2 -> 1 (lever "A" from the FLOPs
discussion). Backbone and neck are UNTOUCHED -- only the head's per-level conv stack
shrinks from two depthwise-separable convs to one.

Measured on this exact architecture at 320x480:
    stock (stacked_convs=2): 0.5025 GMACs / 1.0050 GFLOPs
    this  (stacked_convs=1): 0.4654 GMACs / 0.9308 GFLOPs   (-7.4%)

Why this lever costs you little relative warm-start: PicoDet's head is task-specific
(sized by num_classes), so if your class count differs from COCO's 80 -- as it does
for you -- the head already gets rebuilt fresh, mismatched-shape keys and all, every
time you load the COCO checkpoint. Losing one stacked conv on top of that isn't losing
pretraining you still had; it's a smaller head architecture for the part that was
always starting from scratch. Backbone and neck weights, which ARE genuinely
pretrained and DO transfer, are completely unaffected by this change.

What actually happens when you load a checkpoint trained with the STOCK 2-conv head
into this 1-conv model (verified empirically, not assumed): the first stacked conv's
weights (cls_convs.<level>.0.*) have IDENTICAL shape in both models and load
successfully. The second stacked conv's weights (cls_convs.<level>.1.*) simply don't
exist in this model and are silently dropped -- a genuine partial warm start, not a
crash. LibreYOLO's loader raises on a SHAPE mismatch but only logs (doesn't raise) a
pure key-name mismatch like this one.

Usage -- same flags as your other two scripts:
    python picodet_esnet_headslim.py --data dataset.yaml --imgsz 320 448 \\
        --act leakyrelu --gate sigmoid --device 0

Reload a checkpoint trained with this class (not with plain LibrePICODET, which would
try to rebuild the stock 2-conv head and mismatch):
    model = PicoDetHeadSlim("runs/train/<name>/weights/best.pt", size="s")
"""
import argparse
import inspect
import re

import torch.nn as nn
from libreyolo.models.picodet.model import LibrePICODET
from libreyolo.models.picodet.nn import PicoHead, SIZE_SPEC, LibrePICODETModel

import picodet_letterbox  # noqa: F401  -- letterbox + Paddle-style aug, same as your other scripts
from picodet_activation import ACTS, GATES, set_activation

STACKED_CONVS = 1  # the one number this whole file changes


def _act_candidates():
    """Same defensive lookup as picodet_lcnet_backbone.py's _act_candidates(): read the
    activation name the installed LibrePICODETModel actually uses for its own head,
    since different libreyolo versions spell "hardswish" differently."""
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


def _build_head(neck_ch, head_ch, nb_classes):
    err = None
    for act in _act_candidates():
        try:
            return PicoHead(
                in_channels=neck_ch, num_classes=nb_classes, feat_channels=head_ch,
                stacked_convs=STACKED_CONVS, kernel_size=5, reg_max=7,
                strides=(8, 16, 32, 64), share_cls_reg=True, use_depthwise=True, act=act,
            )
        except ValueError as e:
            if "activation" not in str(e).lower():
                raise
            err = e
    raise RuntimeError(f"none of {_act_candidates()} is accepted by the installed PicoHead") from err


class PicoDetHeadSlimModel(LibrePICODETModel):
    def __init__(self, size="s", nb_classes=80):
        super().__init__(size=size, nb_classes=nb_classes)  # builds stock backbone+neck+head
        spec = SIZE_SPEC[size]
        self.head = _build_head(spec["neck_ch"], spec["head_ch"], nb_classes)  # replace head only


class PicoDetHeadSlim(LibrePICODET):
    """LibrePICODET wrapper that builds PicoDetHeadSlimModel. Backbone/neck are stock,
    so a COCO-pretrained LibrePICODETs.pt loads its backbone+neck normally; only the
    head's extra (now-nonexistent) stacked conv is dropped -- see module docstring."""

    def _init_model(self):
        return PicoDetHeadSlimModel(size=self.size, nb_classes=self.nb_classes)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--weights", default=None, help="starting checkpoint (default: fresh init)")
    p.add_argument("--size", default="s", choices=["s", "m", "l"])
    p.add_argument("--act", default="hswish", choices=sorted(ACTS))
    p.add_argument("--gate", default="default", choices=sorted(GATES))
    p.add_argument("--imgsz", type=int, nargs=2, default=[320, 448], metavar=("H", "W"))
    p.add_argument("--device", default="")
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--lr0", type=float, default=0.01)
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()

    set_activation(args.act, gate=args.gate)  # must run before the model is built
    if args.weights:
        model = PicoDetHeadSlim(args.weights, size=args.size)
    else:
        model = PicoDetHeadSlim(size=args.size)

    n_head = sum(x.numel() for x in model.model.head.parameters()) / 1e6
    n_all = sum(x.numel() for x in model.model.parameters()) / 1e6
    print(f"head params: {n_head:.3f}M | total: {n_all:.3f}M (stock head would be larger)")

    print(model.train(data=args.data, epochs=args.epochs, batch=args.batch, lr0=args.lr0,
                      device=args.device, workers=args.workers, imgsz=tuple(args.imgsz)))
    print(model.val(data=args.data, imgsz=tuple(args.imgsz)))


if __name__ == "__main__":
    main()
