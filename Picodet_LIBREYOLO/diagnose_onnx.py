"""Breaks down an ONNX graph's MACs by OPERATION TYPE (Conv, BatchNormalization,
Softmax, etc.) instead of one aggregate number -- so a discrepancy between two
"identical" exports can be traced to a specific op type, without needing to share
the actual .onnx file. Paste the printed table back for comparison.

Usage:
    python diagnose_onnx.py --onnx picodet_headslim_320x480.onnx
"""
import argparse
from collections import defaultdict

import onnx_tool


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--onnx", required=True)
    args = p.parse_args()

    m = onnx_tool.Model(args.onnx)
    m.graph.shape_infer()
    m.graph.profile()

    by_type = defaultdict(lambda: [0, 0.0])  # op_type -> [count, total_macs]
    for name, node in m.graph.nodemap.items():
        macs = getattr(node, "macs", 0) or 0
        if isinstance(macs, (list, tuple)):
            macs = macs[0]
        by_type[node.op_type][0] += 1
        by_type[node.op_type][1] += macs

    total_macs = sum(v[1] for v in by_type.values())
    print(f"{'Op type':<20} {'Count':>6} {'Total MACs':>14} {'% of total':>10}")
    print("-" * 55)
    for op_type, (count, macs) in sorted(by_type.items(), key=lambda x: -x[1][1]):
        pct = 100 * macs / total_macs if total_macs else 0
        print(f"{op_type:<20} {count:>6} {macs:>14,.0f} {pct:>9.2f}%")
    print("-" * 55)
    print(f"{'TOTAL':<20} {sum(v[0] for v in by_type.values()):>6} {total_macs:>14,.0f}")
    print(f"\nGMACs: {total_macs/1e9:.4f}  |  node count: {len(m.graph.nodemap)}  |  params: {m.graph.params:,}")


if __name__ == "__main__":
    main()
