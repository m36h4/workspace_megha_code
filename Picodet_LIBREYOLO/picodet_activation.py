"""Swap PicoDet's activation (hard-swish -> ReLU6 / SiLU / ...) without editing libreyolo.

Where the activation lives in LibreYOLO's PicoDet (libreyolo/models/picodet/nn.py):
  * ESNet backbone, CSP-PAN neck, PicoHead: all build activations through the module
    function _make_act(name), via ConvBNAct. Names like "hswish" -> nn.Hardswish.
  * SE gates (ESNet backbone only): SELayer = nn.ReLU bottleneck + HSigmoid gate. The ReLU
    is never touched; the gate is only changed if you pass gate="hsigmoid" / "sigmoid".
  * PicoHead has NO hard sigmoid: its convs use the activation above and the final 1x1
    convs output raw logits. The sigmoid on class scores lives in the loss/decoder, not here.
  * act=None means "no activation" (linear); that stays linear.

set_activation() wraps _make_act so that every place that would have built nn.Hardswish
builds your activation instead. It detects hard-swish by the module it returns, not by
name, so it works even if your libreyolo version spells the name differently.

Call it BEFORE the model is created (activations are built in __init__):

    from picodet_activation import set_activation
    set_activation("relu6")
    model = LibreYOLO("LibrePICODETs.pt")     # original ESNet, now with ReLU6
    model.train(...)

Notes:
  * Pretrained weights (COCO ESNet, ImageNet LCNet) were trained with hard-swish. They
    still load (activations have no weights) but accuracy drops until you fine-tune.
  * Activations with parameters (PReLU) would add state-dict keys; not offered here.
  * Multi-GPU (spawned workers) is untested: the patch must also run in each worker.
"""
import torch.nn as nn
import libreyolo.models.picodet.nn as _pnn

ACTS = {
    "hswish": lambda: nn.Hardswish(inplace=True),   # PicoDet default
    "relu": lambda: nn.ReLU(inplace=True),
    "relu6": lambda: nn.ReLU6(inplace=True),        # usually the most quantization/NPU friendly
    "leakyrelu": lambda: nn.LeakyReLU(0.1, inplace=True),
    "silu": lambda: nn.SiLU(inplace=True),          # a.k.a. swish
    "gelu": lambda: nn.GELU(),
    "mish": lambda: nn.Mish(inplace=True),
}

# SE-gate ("hard sigmoid") choices. "default" keeps the model's own gate:
#   ESNet -> libreyolo's HSigmoid, whose output range is [0, 6] (mmcv style, what the COCO
#            weights were trained with); LCNet -> nn.Hardsigmoid, range [0, 1].
GATES = {
    "default": None,
    "hsigmoid": lambda: nn.Hardsigmoid(),   # standard hard sigmoid, [0, 1]
    "sigmoid": lambda: nn.Sigmoid(),        # [0, 1], smooth
}

_current = "hswish"
_gate = "default"


def make_act() -> nn.Module:
    """A fresh instance of the currently selected activation (used by custom backbones)."""
    return ACTS[_current]()


def make_gate() -> nn.Module:
    """SE gate for custom backbones (LCNet). Default is nn.Hardsigmoid."""
    return nn.Hardsigmoid() if _gate == "default" else GATES[_gate]()


def current_activation() -> str:
    return _current


def set_activation(name: str, gate: str = "default") -> None:
    """name: activation replacing hard-swish. gate: SE-gate replacing the hard sigmoid."""
    global _current, _gate
    name, gate = name.lower(), gate.lower()
    if name not in ACTS:
        raise ValueError(f"unknown activation {name!r}; choose from {sorted(ACTS)}")
    if gate not in GATES:
        raise ValueError(f"unknown gate {gate!r}; choose from {sorted(GATES)}")
    if not hasattr(_pnn, "_orig_make_act"):          # keep the original, patch only once
        _pnn._orig_make_act = _pnn._make_act
    orig = _pnn._orig_make_act

    def _patched_make_act(act):
        module = orig(act)
        return make_act() if isinstance(module, nn.Hardswish) else module

    _pnn._make_act = _patched_make_act
    _current = name

    # ESNet's SELayer builds its gate with the module-level name HSigmoid() at init time.
    if not hasattr(_pnn, "_orig_HSigmoid"):
        _pnn._orig_HSigmoid = _pnn.HSigmoid
    _pnn.HSigmoid = _pnn._orig_HSigmoid if gate == "default" else GATES[gate]
    _gate = gate
