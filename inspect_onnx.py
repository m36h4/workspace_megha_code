"""
PicoDet ONNX — postprocessing inspector + minimal inference/NMS demo.

Part A: statically inspects the graph to show you exactly which ops
        run between the raw head convs and the final `output` tensor
        (DFL decode, anchor decode, concat) — no image needed.

Part B: runs the model on a dummy (or real) image and applies the
        confidence threshold + NMS step that happens OUTSIDE the onnx
        graph, so you can see the full pipeline end-to-end.

Usage:
    pip install onnx onnxruntime numpy opencv-python --break-system-packages
    python picodet_postprocess_inspect.py --model pytorch_picodet_sim.onnx
    python picodet_postprocess_inspect.py --model pytorch_picodet_sim.onnx --image path/to/img.jpg
"""

import argparse
import numpy as np
import onnx
from onnx import numpy_helper


# ----------------------------------------------------------------------
# Part A — static graph inspection
# ----------------------------------------------------------------------
def inspect_graph(model_path):
    m = onnx.load(model_path)
    g = m.graph

    print("=" * 70)
    print("MODEL INFO")
    print("=" * 70)
    print("Producer:", m.producer_name, m.producer_version)
    print("Opset:", [(o.domain, o.version) for o in m.opset_import])
    for i in g.input:
        dims = [d.dim_value or d.dim_param for d in i.type.tensor_type.shape.dim]
        print("Input :", i.name, dims)
    for o in g.output:
        dims = [d.dim_value or d.dim_param for d in o.type.tensor_type.shape.dim]
        print("Output:", o.name, dims)

    # Find the gfl_cls conv weight shapes -> tells you num_classes & reg_max
    print("\n" + "=" * 70)
    print("HEAD CONFIG (derived from weight shapes)")
    print("=" * 70)
    gfl_cls_shape = None
    for init in g.initializer:
        if "gfl_cls" in init.name and "weight" in init.name:
            gfl_cls_shape = list(init.dims)
            print(f"{init.name}: {gfl_cls_shape}")

    # Find the actual Split sizes used to break gfl_cls output into
    # (num_classes) and (4*(reg_max+1))
    split_sizes = None
    for n in g.node:
        if n.op_type == "Split" and n.name.startswith("/head/Split"):
            # second input is the split-sizes initializer (opset13 style)
            if len(n.input) > 1:
                for init in g.initializer:
                    if init.name == n.input[1]:
                        split_sizes = numpy_helper.to_array(init)
            print(f"{n.name} splits channel dim into: {split_sizes}")
            break

    if split_sizes is not None:
        num_classes = int(split_sizes[0])
        reg_channels = int(split_sizes[1])
        reg_max = reg_channels // 4 - 1
        print(f"\n=> num_classes = {num_classes}")
        print(f"=> reg_max     = {reg_max}  (4 sides x {reg_max+1} bins = {reg_channels} channels)")

    # Walk backward from the final output to show the decode chain once
    print("\n" + "=" * 70)
    print("POST-HEAD DECODE CHAIN (backward trace from `output`, one branch)")
    print("=" * 70)
    name_to_node = {}
    for n in g.node:
        for o in n.output:
            name_to_node[o] = n

    def trace(name, depth=0, maxdepth=12, seen=None):
        if seen is None:
            seen = set()
        if name in seen or depth > maxdepth:
            return
        seen.add(name)
        n = name_to_node.get(name)
        if n is None:
            print("  " * depth + f"[input/initializer] {name}")
            return
        print("  " * depth + f"{n.op_type:12s} ({n.name})")
        for i in n.input:
            trace(i, depth + 1, maxdepth, seen)

    # Trace only the first FPN-level branch (Concat_4) for readability
    trace("/head/Concat_4_output_0", maxdepth=8)

    print("\nInterpretation of this chain (typical PicoDet):")
    print("  gfl_cls Conv -> Split[cls, reg] -> ")
    print("    cls branch:  Sigmoid -> Transpose -> Reshape          (per-class scores)")
    print("    reg branch:  Reshape -> Softmax -> Mul -> ReduceSum   (DFL: expected value")
    print("                 over reg_max+1 bins per side, per anchor)")
    print("  Then Sub/Add combine (anchor_center, distance) -> (x1,y1,x2,y2)")
    print("  Finally Concat[boxes(4), scores(num_classes)] per FPN level,")
    print("  then Concat across all FPN levels -> final `output` [1, num_anchors, 4+num_classes]")


# ----------------------------------------------------------------------
# Part B — run inference + apply the postprocessing NOT in the graph
# (confidence filtering + NMS)
# ----------------------------------------------------------------------
def nms(boxes, scores, iou_thresh=0.5):
    """Plain NumPy NMS. boxes: [N,4] x1y1x2y2, scores: [N]"""
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        inds = np.where(iou <= iou_thresh)[0]
        order = order[inds + 1]
    return keep


def run_inference(model_path, image_path=None, conf_thresh=0.3, iou_thresh=0.5):
    import onnxruntime as ort

    sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0]
    _, c, h, w = inp.shape

    if image_path:
        import cv2
        img = cv2.imread(image_path)
        img = cv2.resize(img, (w, h))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = (img - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]  # ImageNet norm; adjust if trained differently
        blob = img.transpose(2, 0, 1)[None].astype(np.float32)
    else:
        blob = np.random.rand(1, c, h, w).astype(np.float32)

    out = sess.run(None, {inp.name: blob})[0]  # [1, N, 4+num_classes]
    out = out[0]
    boxes = out[:, :4]
    scores = out[:, 4:]  # per-class scores, already sigmoid-activated in-graph

    num_classes = scores.shape[1]
    print(f"\nRaw output shape: {out.shape}  (num_classes={num_classes})")

    final = []
    for cls_id in range(num_classes):
        cls_scores = scores[:, cls_id]
        mask = cls_scores > conf_thresh
        if not mask.any():
            continue
        b, s = boxes[mask], cls_scores[mask]
        keep = nms(b, s, iou_thresh)
        for k in keep:
            final.append((cls_id, s[k], *b[k]))

    print(f"Detections after conf>{conf_thresh} + NMS(iou>{iou_thresh}): {len(final)}")
    for cls_id, score, x1, y1, x2, y2 in final:
        print(f"  class={cls_id} score={score:.3f} box=({x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f})")

    return final


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--image", default=None, help="optional real image to run through the model")
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--iou", type=float, default=0.5)
    args = ap.parse_args()

    inspect_graph(args.model)

    print("\n" + "=" * 70)
    print("RUNNING INFERENCE" + (" (dummy random input)" if not args.image else f" on {args.image}"))
    print("=" * 70)
    run_inference(args.model, args.image, args.conf, args.iou)
