"""Run this after extracting picodet_update.zip. Prints PASS/FAIL per check."""
import collections
from pathlib import Path

here = Path(__file__).resolve().parent
ok = True


def check(name, cond, detail=""):
    global ok
    print(("PASS" if cond else "FAIL"), "-", name, detail)
    ok = ok and cond


src = (here / "picodet_esnet_headslim.py").read_text()
check("headslim file has the activation + gate fix", "_force_activation" in src and "_force_gate" in src)

import picodet_activation as pa
check("default activation is leakyrelu", pa._current == "leakyrelu", f"(found {pa._current})")
check("default gate is sigmoid", pa._gate == "sigmoid", f"(found {pa._gate})")

from picodet_esnet_headslim import PicoDetHeadSlim
m = PicoDetHeadSlim(size="s", nb_classes=1)
c = dict(collections.Counter(
    type(x).__name__ for x in m.model.modules()
    if not list(x.children()) and any(k in type(x).__name__.lower() for k in ("relu", "silu", "sigmoid", "swish"))))
check("model built with NO flags has 79 LeakyReLU, 13 ReLU (SE), 13 Sigmoid (SE gate), nothing else",
      c == {"LeakyReLU": 79, "ReLU": 13, "Sigmoid": 13}, f"(found {c})")

print("\nALL GOOD" if ok else "\nSOMETHING IS WRONG - an old file is still being used")
