"""Static audit of an ONNX file against every problem the client (Ohata-san) reported on
the Teams thread "PicoDet weight files for hardware compatibility testing".

It can only check what is visible in the file. It CANNOT tell you whether the client's
compiler accepts a given op pattern -- for that it prints the facts (ops, opset, concat
axes) so they can be compared with what the compiler is known to accept.

Usage:
    python check_client_compat.py --onnx picodet_v3.onnx
    python check_client_compat.py --onnx picodet_v3.onnx --supported client_supported_ops.txt
(--supported: text file, one ONNX op name per line, copied from the client's layer-limitation slide)
"""
import argparse
from collections import Counter

import onnx
from onnx import shape_inference

# op -> what happened on the thread
THREAD_OPS = {
    "NonMaxSuppression": "8/5  compiler does not support NonMaxSuppression",
    "MatMul": "8/6  compile error around MatMul",
    "HardSwish": "8/17 even a tiny Conv+HardSwish test model failed on the compiler",
    "HardSigmoid": "8/17 hard-sigmoid is banned together with hard-swish",
}
WATCH_OPS = {
    "Shape": "shape-computation op (the 8/3 Gather rank error came from graphs like this)",
    "ConstantOfShape": "shape-computation op", "NonZero": "data-dependent shapes", "Loop": "control flow",
    "If": "control flow", "Gemm": "same family as MatMul",
    "Clip": "what a hard-sigmoid exports as (Add/Div/Clip); also ReLU6",
}


def dims_of(vi):
    t = vi.type.tensor_type
    return [d.dim_value if d.HasField("dim_value") else (d.dim_param or "?") for d in t.shape.dim]


def rank_map(model):
    r = {}
    for vi in list(model.graph.input) + list(model.graph.output) + list(model.graph.value_info):
        if vi.type.tensor_type.HasField("shape"):
            r[vi.name] = len(vi.type.tensor_type.shape.dim)
    for init in model.graph.initializer:
        r[init.name] = len(init.dims)
    for n in model.graph.node:
        if n.op_type == "Constant":
            for a in n.attribute:
                if a.name == "value":
                    r[n.output[0]] = len(a.t.dims)
    return r


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--onnx", required=True)
    p.add_argument("--supported", default=None)
    args = p.parse_args()

    model = onnx.load(args.onnx)
    fails, watches = [], []

    def line(level, msg):
        print(f"{level:<6} {msg}")
        (fails if level == "FAIL" else watches if level == "WATCH" else []).append(msg)

    ops = Counter(n.op_type for n in model.graph.node)
    print(f"{args.onnx}: {len(model.graph.node)} nodes, opset {[(o.domain or 'ai.onnx', o.version) for o in model.opset_import]}")
    print("ops:", ", ".join(f"{k} x{v}" for k, v in sorted(ops.items(), key=lambda x: -x[1])), "\n")

    # 1. ops that failed on the thread
    for op, why in THREAD_OPS.items():
        if ops.get(op):
            line("FAIL", f"{op} x{ops[op]} present -- {why}")
        else:
            line("PASS", f"no {op} -- ({why})")
    for op, why in WATCH_OPS.items():
        if ops.get(op):
            line("WATCH", f"{op} x{ops[op]} present -- {why}")

    # 2. the 8/3 error: ONNX shape inference / Gather with a rank-0 data tensor
    try:
        inferred = shape_inference.infer_shapes(model, check_type=True, strict_mode=True)
        line("PASS", "strict ONNX shape inference succeeds (the 8/3 'ShapeInferenceError' class)")
    except Exception as e:
        inferred = model
        line("FAIL", f"strict ONNX shape inference fails: {str(e).strip().splitlines()[0][:160]}")
    ranks = rank_map(inferred)
    bad, unknown, n_gather = [], 0, 0
    for n in inferred.graph.node:
        if n.op_type == "Gather":
            n_gather += 1
            r = ranks.get(n.input[0])
            if r is None:
                unknown += 1
            elif r < 1:
                bad.append(n.name or n.output[0])
    if bad:
        line("FAIL", f"{len(bad)} Gather node(s) with a rank-0 data tensor, e.g. {bad[:3]}")
    elif n_gather:
        line("PASS" if not unknown else "WATCH", f"all {n_gather - unknown}/{n_gather} Gather data tensors have rank >= 1"
             + (f" ({unknown} of unknown rank)" if unknown else ""))

    # 3. inputs / outputs
    for vi in model.graph.input:
        d = dims_of(vi)
        line("PASS" if all(isinstance(x, int) for x in d) else "WATCH", f"input {vi.name}: {d}"
             + ("" if all(isinstance(x, int) for x in d) else "  (dynamic dims)"))
    for vi in model.graph.output:
        d = dims_of(vi)
        print(f"INFO   output {vi.name}: {d}"
              + (f"  -> last dim = 4 box coords + {d[-1] - 4} class score(s)" if isinstance(d[-1], int) else ""))

    # 4. Reshape/Concat -- the pattern the client's compiler struggled with (8/6, 8/11)
    static = {i.name for i in model.graph.initializer} | {n.output[0] for n in model.graph.node if n.op_type == "Constant"}
    dyn_reshape = sum(1 for n in model.graph.node if n.op_type == "Reshape" and n.input[1] not in static)
    print(f"INFO   Reshape x{ops.get('Reshape', 0)} ({dyn_reshape} with a computed, non-constant target shape)")
    cc = Counter()
    for n in inferred.graph.node:
        if n.op_type == "Concat":
            axis = next((a.i for a in n.attribute if a.name == "axis"), None)
            cc[(axis, ranks.get(n.input[0]), len(n.input))] += 1
    for (axis, rank, nin), c in sorted(cc.items(), key=lambda x: str(x[0])):
        print(f"INFO   Concat x{c}: axis={axis}, rank={rank}, {nin} inputs")
    if any(axis in (1, -2) and nin >= 4 for (axis, rank, nin) in cc):
        line("WATCH", "a 4-input Concat on axis 1 of rank-3 tensors (B x N x C) exists -- the same 'Bx100x4, Bx25x4, "
                      "Bx400x4, Bx1600x4' merge pattern the client asked to restructure on 8/6. Not known to be a "
                      "problem for this graph; only their compiler can say.")

    # 5. client's supported-op list (8/3: 'some layers are not in supported list')
    if args.supported:
        sup = {l.strip() for l in open(args.supported) if l.strip() and not l.startswith("#")}
        missing = sorted(set(ops) - sup)
        line("FAIL" if missing else "PASS", f"ops not in your supported list: {missing or 'none'}")
    else:
        print("INFO   (pass --supported <file> with the client's supported-op list to check op-by-op)")

    print(f"\nRESULT: {len(fails)} FAIL, {len(watches)} WATCH")


if __name__ == "__main__":
    main()
