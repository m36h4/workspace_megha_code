"""Tiny PicoDet-ESNet for latency: strides 8/16 only, scaled backbone, fewer stride-16 blocks,
narrow neck, kernel 3. LeakyReLU + sigmoid gate. From scratch (shapes differ from any checkpoint),
so distill from your current best model (distill_picodet.py style) or train long.

    python picodet_esnet_tiny.py --profile-only --preset t1
    python picodet_esnet_tiny.py --data ds.yaml --preset t1 --device 0

Reload:  from picodet_esnet_tiny import load_tiny; m = load_tiny("best.pt", preset="t1")
Export:  python export_onnx.py --arch esnet_tiny --tiny-preset t1 --head-style raw ...
"""
import argparse
import torch
import torch.nn as nn
import libreyolo.training.trainer as _trainer
from libreyolo.models.picodet.model import LibrePICODET
from libreyolo.models.picodet.nn import CSPPAN, ESNet, PicoHead, SIZE_SPEC, LibrePICODETModel

import picodet_letterbox  # noqa: F401
from picodet_activation import ACTS, GATES, set_activation
from picodet_esnet_headslim import _act_candidates, _force_activation, _force_gate

STRIDES = (8, 16)
_trainer.RECTANGULAR_TRAINING_FAMILIES["picodet"] = 16
# scale = ESNet width multiplier (stock "s" is 0.75), s16 = number of ESBlocks kept after the
# stride-16 downsample block (stock has 6), neck = neck/head width, k = depthwise kernel
PRESETS = {
    "t0": dict(scale=0.5,   s16=4, neck=48, k=3),
    "t1": dict(scale=0.375, s16=3, neck=48, k=3),
    "t2": dict(scale=0.375, s16=2, neck=32, k=3),
    "t3": dict(scale=0.25,  s16=2, neck=32, k=3),
}


class ESNetTiny(nn.Module):
    def __init__(self, scale, s16):
        super().__init__()
        key = f"tiny_{scale}"
        if key not in ESNet.ARCH:
            ESNet.ARCH[key] = {"scale": scale, "ratios": ESNet.ARCH["s"]["ratios"]}
        e = ESNet(model_size=key)
        self.idx = (2, 3 + s16)                       # last stride-8 block, last kept stride-16 block
        self.conv1, self.max_pool = e.conv1, e.max_pool
        self.blocks = nn.ModuleList(list(e.blocks)[: self.idx[-1] + 1])
        self.out_channels = (e.out_channels[0], e.out_channels[1])

    def forward(self, x):
        x = self.max_pool(self.conv1(x))
        outs = []
        for i, b in enumerate(self.blocks):
            x = b(x)
            if i in self.idx:
                outs.append(x)
        return outs


class PicoDetTinyModel(LibrePICODETModel):
    def __init__(self, size="s", nb_classes=80, scale=0.375, s16=3, neck=48, k=3, reg_max=7):
        super().__init__(size=size, nb_classes=nb_classes)
        self.backbone = ESNetTiny(scale, s16)
        err = None
        for act in _act_candidates():
            try:
                self.neck = CSPPAN(in_channels=self.backbone.out_channels, out_channels=neck, kernel_size=k,
                                   num_features=3, expansion=1.0, num_csp_blocks=1, use_depthwise=True, act=act)
                self.head = PicoHead(in_channels=neck, num_classes=nb_classes, feat_channels=neck,
                                     stacked_convs=1, kernel_size=k, reg_max=reg_max, strides=STRIDES,
                                     share_cls_reg=True, use_depthwise=True, act=act)
                break
            except ValueError as e:
                if "activation" not in str(e).lower():
                    raise
                err = e
        else:
            raise RuntimeError("no activation name accepted") from err
        for part in (self.backbone, self.neck):
            for m in part.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)
        for part in (self.backbone, self.neck, self.head):
            _force_activation(part)
        _force_gate(self.backbone)


class PicoDetTiny(LibrePICODET):
    scale, s16, neck, k, reg_max = 0.375, 3, 48, 3, 7     # reg_max=0 -> "no GFL": 4 direct distances

    def _init_model(self):
        return PicoDetTinyModel(self.size, self.nb_classes, self.scale, self.s16, self.neck, self.k, self.reg_max)

    def _get_available_layers(self):
        b = self.model.backbone
        return {"backbone_conv1": b.conv1, "backbone_blocks": b.blocks,
                "neck": self.model.neck, "head": self.model.head}


def _neck_distill_config(self):
    """Distillation tap points for the 2-level (strides 8/16) models: tiny student AND the
    small-stride teacher. Channels are read from the real neck, so a 64-wide teacher and a
    48-wide student work (the distiller adds a projection when widths differ)."""
    ch = self.model.neck.trans[0].conv.out_channels
    return {"tap_points": ["neck.top_down_blocks.0", "neck.bottom_up_blocks.0"],
            "channels": [ch, ch], "strides": [8, 16]}


PicoDetTiny.get_distill_config = _neck_distill_config


def warm_start_tiny(model, path):
    """Loads every same-shape tensor from a trained tiny checkpoint (e.g. Exp-3 best.pt) into a fresh
    model, so a NEW training run (new LR / schedule / activation) starts from those weights."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    src = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
    own = model.model.state_dict()
    take = {k: v for k, v in src.items() if k in own and own[k].shape == v.shape}
    model.model.load_state_dict(take, strict=False)
    print(f"  warm start: {len(take)}/{len(own)} tensors from {path}")
    if len(take) < 0.95 * len(own):
        print("  WARNING: fewer than 95% of tensors matched -- wrong --preset?")
    return model


def _install_teacher(teacher_path, act, gate):
    """The trainer builds the distillation teacher with LibreYOLO(<path>), which would build the
    WRONG (4-level) PicoDet for a strides-8/16 checkpoint. Return the right class instead, built
    with the activation the teacher was trained with."""
    import libreyolo.models as lm
    from picodet_activation import activation_scope
    from picodet_esnet_small import load_small, PicoDetSmallStride
    PicoDetSmallStride.get_distill_config = _neck_distill_config
    real = lm.LibreYOLO

    def factory(*args, **kwargs):
        if args and str(args[0]) == str(teacher_path):
            with activation_scope(act, gate):
                t = load_small(teacher_path)
            print(f"[distill] teacher {type(t).__name__} neck={t.model.neck.trans[0].conv.out_channels} act={act}")
            return t
        return real(*args, **kwargs)

    lm.LibreYOLO = factory


def _install_loss(cls_w=1.0, bbox_w=2.0, dfl_w=0.25, ota_iou_w=6.0, center_radius=2.5,
                  candidate_topk=10, min_k=1):
    """Overrides PicoDet's loss / SimOTA assigner settings (LibreYOLO does not expose them in train()).
    Defaults reproduce the stock loss exactly. min_k is the minimum number of positive anchors per
    ground-truth box: stock is 1, which leaves a small ball with a single positive prior early in
    training (its predicted-box IoUs are low, so dynamic-k rounds down to 1)."""
    import libreyolo.models.picodet.trainer as _pt
    from libreyolo.models.picodet.loss import PICODETLoss, SimOTAAssigner

    class Assigner(SimOTAAssigner):
        def _dynamic_k_matching(self, cost, pairwise_ious, num_gts, valid_mask):
            if min_k <= 1:
                return super()._dynamic_k_matching(cost, pairwise_ious, num_gts, valid_mask)
            matching = torch.zeros_like(cost, dtype=torch.uint8)
            topk = min(self.candidate_topk, pairwise_ious.shape[0])
            topk_ious, _ = torch.topk(pairwise_ious, topk, dim=0)
            ks = topk_ious.sum(0).int().clamp(min=min_k, max=cost.shape[0])
            for g in range(num_gts):
                _, idx = torch.topk(cost[:, g], k=int(ks[g].item()), largest=False)
                matching[idx, g] = 1
            multi = matching.sum(1) > 1
            if multi.any():
                _, best = cost[multi].min(dim=1)
                matching[multi] = 0
                matching[multi, best] = 1
            fg = matching.sum(1) > 0
            valid_mask[valid_mask.clone()] = fg
            return (matching * pairwise_ious).sum(1)[fg], matching[fg].argmax(1)

    def build(**kw):
        loss = PICODETLoss(cls_loss_weight=cls_w, bbox_loss_weight=bbox_w, dfl_loss_weight=dfl_w,
                           sim_ota_iou_weight=ota_iou_w, **kw)
        loss.assigner = Assigner(center_radius=center_radius, candidate_topk=candidate_topk, iou_weight=ota_iou_w)
        return loss

    _pt.PICODETLoss = build
    print(f"[loss] cls_w={cls_w} bbox_w={bbox_w} dfl_w={dfl_w} ota_iou_w={ota_iou_w} "
          f"center_radius={center_radius} candidate_topk={candidate_topk} min_k={min_k}")


def _reg_max_from_checkpoint(path):
    """GFL head: last conv has nc + 4*(reg_max+1) channels. Detect reg_max (7 = GFL, 0 = no GFL)."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    sd = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
    key = next(k for k in sd if k.startswith("head.gfl_cls.0") and k.endswith("weight"))
    out = sd[key].shape[0]
    nc = ckpt.get("nc", None) if isinstance(ckpt, dict) else None
    for rm in (7, 0, 3, 15):        # nc = out - 4*(rm+1) must be >= 1
        if out - 4 * (rm + 1) >= 1 and (nc is None or out - 4 * (rm + 1) == nc):
            return rm
    return 7


def load_tiny(path, size="s", preset="t1"):
    p = PRESETS[preset]
    return PicoDetTiny(path, size=size, scale=p["scale"], s16=p["s16"], neck=p["neck"], k=p["k"],
                       reg_max=_reg_max_from_checkpoint(path))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data")
    ap.add_argument("--preset", default="t1", choices=sorted(PRESETS))
    ap.add_argument("--act", default="leakyrelu", choices=sorted(ACTS))
    ap.add_argument("--gate", default="sigmoid", choices=sorted(GATES))
    ap.add_argument("--imgsz", type=int, nargs=2, default=[320, 480])
    ap.add_argument("--device", default="")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr0", type=float, default=0.1)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--init-weights", default=None, metavar="BEST_PT",
                    help="start a NEW run (fresh LR/schedule) from trained tiny weights, e.g. Exp-3 best.pt; "
                         "needs the same --preset")
    ap.add_argument("--teacher", default=None, metavar="CKPT",
                    help="distill from a strides-8/16 checkpoint (e.g. Exp-2.1.1, neck 64); "
                         "teacher activation set by --teacher-act/--teacher-gate")
    ap.add_argument("--teacher-act", default="leakyrelu", choices=sorted(ACTS))
    ap.add_argument("--teacher-gate", default="sigmoid", choices=sorted(GATES))
    ap.add_argument("--distill-loss", default="mgd", choices=["mgd", "cwd", "feat_mse"])
    ap.add_argument("--dis", type=float, default=None, help="global distillation weight (default per loss)")
    # --- loss / assigner (defaults = stock PicoDet) ---
    ap.add_argument("--cls-w", type=float, default=1.0, help="varifocal (classification) loss weight")
    ap.add_argument("--bbox-w", type=float, default=2.0, help="GIoU loss weight")
    ap.add_argument("--dfl-w", type=float, default=0.25, help="distribution focal loss weight")
    ap.add_argument("--ota-iou-w", type=float, default=6.0, help="SimOTA IoU cost weight")
    ap.add_argument("--center-radius", type=float, default=2.5, help="SimOTA candidate radius, in strides")
    ap.add_argument("--candidate-topk", type=int, default=10)
    ap.add_argument("--min-k", type=int, default=1, help="minimum positive anchors per box (stock 1)")
    # --- schedule (defaults = LibreYOLO PicoDet) ---
    ap.add_argument("--weight-decay", type=float, default=4e-5)
    ap.add_argument("--warmup-epochs", type=int, default=1)
    ap.add_argument("--min-lr-ratio", type=float, default=0.0, help="final LR = lr0 x this")
    ap.add_argument("--resume", default=None, metavar="LAST_PT",
                    help="continue an interrupted run from its last.pt (same --preset/--epochs/--batch/--lr0 as the original run)")
    ap.add_argument("--no-gfl", action="store_true",
                    help="no Generalized Focal Loss head: 4 direct box distances instead of a 4x8 distribution "
                         "(build/export only for now -- training needs the matching loss)")
    ap.add_argument("--save-untrained", default=None, metavar="OUT_PT",
                    help="save an untrained checkpoint (for a hardware feasibility check) and exit")
    ap.add_argument("--profile-only", action="store_true")
    a = ap.parse_args()
    set_activation(a.act, gate=a.gate)
    p = PRESETS[a.preset]
    if a.resume:
        model = load_tiny(a.resume, size="s", preset=a.preset)      # builds the tiny net, loads last.pt
    else:
        model = PicoDetTiny(size="s", nb_classes=1, scale=p["scale"], s16=p["s16"], neck=p["neck"], k=p["k"],
                            reg_max=0 if a.no_gfl else 7)
    if a.save_untrained:
        torch.save({"model": model.model.state_dict(), "nc": 1, "reg_max": 0 if a.no_gfl else 7,
                    "act": a.act, "gate": a.gate, "preset": a.preset}, a.save_untrained)
        print(f"saved untrained {a.preset} (act={a.act}, gate={a.gate}, no_gfl={a.no_gfl}) -> {a.save_untrained}")
        return
    if a.no_gfl and not a.profile_only:
        raise SystemExit("--no-gfl training is not supported yet (the loss still expects the GFL distribution). "
                         "Use --save-untrained / --profile-only for the hardware check.")
    if a.init_weights:
        warm_start_tiny(model, a.init_weights)
    if a.profile_only:
        from torchinfo import summary
        s = summary(model.model.eval(), input_size=(1, 3, *a.imgsz), verbose=0)
        print(f"{a.preset} {p}: {s.total_mult_adds/1e9:.4f} GMACs, {2*s.total_mult_adds/1e9:.4f} GFLOPs, {s.total_params:,} params")
        return
    _install_loss(a.cls_w, a.bbox_w, a.dfl_w, a.ota_iou_w, a.center_radius, a.candidate_topk, a.min_k)
    extra = dict(weight_decay=a.weight_decay, warmup_epochs=a.warmup_epochs, min_lr_ratio=a.min_lr_ratio)
    if a.teacher:
        _install_teacher(a.teacher, a.teacher_act, a.teacher_gate)
        extra.update(distill_model=a.teacher, distill_loss_type=a.distill_loss)
        if a.dis is not None:
            extra["dis"] = a.dis
    print(model.train(data=a.data, epochs=a.epochs, batch=a.batch, lr0=a.lr0, device=a.device,
                      workers=a.workers, imgsz=tuple(a.imgsz), resume=bool(a.resume), **extra))
    print(model.val(data=a.data, imgsz=tuple(a.imgsz)))


if __name__ == "__main__":
    main()
