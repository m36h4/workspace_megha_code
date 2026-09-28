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


def _force_activation(module):
    """Walks ``module`` and replaces every activation submodule with a fresh
    instance of the CURRENTLY selected activation (picodet_activation.make_act()).

    Why this exists instead of trusting picodet_activation.py's global patch alone:
    that patch only intercepts calls that would have built nn.Hardswish specifically.
    Confirmed on a real install: this library's own activation construction can
    resolve directly to nn.SiLU (not disguised as Hardswish at all), in which case
    the patch's isinstance(module, nn.Hardswish) check never fires and --act is
    silently ignored. This function sidesteps that entirely: it doesn't care what
    the module WAS built as, or why -- it just makes every activation instance,
    unconditionally, match what the user actually asked for.

    Confirmed necessary for backbone AND neck, not just the head: on a real trained
    checkpoint, a .pt-side module-type breakdown showed 79 nn.SiLU instances despite
    --act leakyrelu -- far more than the ~8 activation sites in a slimmed head alone,
    meaning backbone/neck were affected too.
    """
    from picodet_activation import make_act

    # nn.ReLU is deliberately NOT in this list: ESNet's SE blocks use a plain ReLU
    # bottleneck that must stay ReLU whatever --act is (picodet_activation.py leaves it alone too).
    act_types = (nn.Hardswish, nn.SiLU, nn.LeakyReLU, nn.ReLU6, nn.GELU, nn.Mish)
    for sub in module.modules():
        for name, child in sub.named_children():
            is_std = isinstance(child, act_types)
            # Also catch a CUSTOM activation class that isinstance() can't see: ConvBNAct-style
            # blocks keep their activation in an attribute literally named "act" (Identity when
            # the block has no activation, which must stay Identity).
            is_named_act = (name == "act" and not isinstance(child, nn.Identity)
                            and not any(True for _ in child.children()))
            if is_std or is_named_act:
                setattr(sub, name, make_act())


def _force_gate(module):
    """Replaces every SE-gate module (any leaf whose class name contains "sigmoid",
    e.g. libreyolo's HSigmoid, nn.Hardsigmoid, nn.Sigmoid) with the selected gate
    (picodet_activation.make_gate(), nn.Sigmoid by default). Needed because ESNet's
    SELayer builds its gate from libreyolo's own HSigmoid unless set_activation()
    was called first -- and that is a hard-sigmoid, which the client bans."""
    from picodet_activation import make_gate

    for sub in module.modules():
        for name, child in sub.named_children():
            if "sigmoid" in type(child).__name__.lower() and not any(True for _ in child.children()):
                setattr(sub, name, make_gate())


def _build_head(neck_ch, head_ch, nb_classes):
    """Builds PicoHead, then forces every activation submodule to the currently
    selected activation via _force_activation() -- see that function's docstring.
    The exact name tried here is irrelevant to the final result (every activation
    instance gets overwritten immediately after); it only needs to be A name the
    installed PicoHead accepts without raising.
    """
    err = None
    head = None
    for act in _act_candidates():
        try:
            head = PicoHead(
                in_channels=neck_ch, num_classes=nb_classes, feat_channels=head_ch,
                stacked_convs=STACKED_CONVS, kernel_size=5, reg_max=7,
                strides=(8, 16, 32, 64), share_cls_reg=True, use_depthwise=True, act=act,
            )
            break
        except ValueError as e:
            if "activation" not in str(e).lower():
                raise
            err = e
    if head is None:
        raise RuntimeError(f"none of {_act_candidates()} is accepted by the installed PicoHead") from err
    _force_activation(head)
    return head


class PicoDetHeadSlimModel(LibrePICODETModel):
    def __init__(self, size="s", nb_classes=80):
        super().__init__(size=size, nb_classes=nb_classes)  # builds stock backbone+neck+head
        spec = SIZE_SPEC[size]
        self.head = _build_head(spec["neck_ch"], spec["head_ch"], nb_classes)  # replace head only
        _force_activation(self.backbone)  # belt-and-braces -- see _force_activation's docstring;
        _force_activation(self.neck)      # confirmed necessary on at least one real install.
        _force_gate(self.backbone)        # SE gates: never the library's hard-sigmoid
        _force_activation(self.backbone)  # belt-and-braces: see _force_activation's docstring
        _force_activation(self.neck)      # -- confirmed necessary on at least one real install,
                                          # where backbone/neck activations were ALSO silently
                                          # wrong for the same reason the head was.


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
    p.add_argument("--act", default="leakyrelu", choices=sorted(ACTS))
    p.add_argument("--gate", default="sigmoid", choices=sorted(GATES))
    p.add_argument("--imgsz", type=int, nargs=2, default=[320, 448], metavar=("H", "W"))
    p.add_argument("--device", default="")
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr0", type=float, default=0.1)
    p.add_argument("--workers", type=int, default=8)
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
