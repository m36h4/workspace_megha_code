#!/usr/bin/env python3

import argparse
from collections import Counter

import onnx
import onnxruntime as ort
from onnx import numpy_helper, shape_inference


def get_shape(value):
    dims = []
    for d in value.type.tensor_type.shape.dim:
        if d.dim_param:
            dims.append(d.dim_param)
        elif d.dim_value:
            dims.append(d.dim_value)
        else:
            dims.append("?")
    return dims


def dtype_name(dtype):
    return onnx.TensorProto.DataType.Name(dtype)


def main():

    parser = argparse.ArgumentParser(description="Inspect ONNX model")
    parser.add_argument("model", help="Path to ONNX model")
    args = parser.parse_args()

    print("=" * 120)
    print("MODEL :", args.model)
    print("=" * 120)

    model = onnx.load(args.model)

    try:
        model = shape_inference.infer_shapes(model)
        print("Shape inference : SUCCESS\n")
    except Exception as e:
        print("Shape inference FAILED")
        print(e)
        print()

    ##############################################################
    # Tensor shape dictionary
    ##############################################################

    tensor_shapes = {}

    for t in (
        list(model.graph.input)
        + list(model.graph.value_info)
        + list(model.graph.output)
    ):
        tensor_shapes[t.name] = get_shape(t)

    ##############################################################
    # Initializers
    ##############################################################

    initializer = {}

    for t in model.graph.initializer:
        initializer[t.name] = numpy_helper.to_array(t)

    ##############################################################
    # Model Inputs
    ##############################################################

    print("=" * 120)
    print("MODEL INPUTS")
    print("=" * 120)

    for inp in model.graph.input:
        print(inp.name)
        print(" Shape :", get_shape(inp))
        print(" DType :", dtype_name(inp.type.tensor_type.elem_type))
        print()

    ##############################################################
    # Model Outputs
    ##############################################################

    print("=" * 120)
    print("MODEL OUTPUTS")
    print("=" * 120)

    for out in model.graph.output:
        print(out.name)
        print(" Shape :", get_shape(out))
        print(" DType :", dtype_name(out.type.tensor_type.elem_type))
        print()

    ##############################################################
    # Operator counts
    ##############################################################

    print("=" * 120)
    print("OPERATOR COUNTS")
    print("=" * 120)

    cnt = Counter(n.op_type for n in model.graph.node)

    for op in sorted(cnt):
        print(f"{op:20} {cnt[op]}")

    ##############################################################
    # Reshape nodes
    ##############################################################

    print("\n")
    print("=" * 120)
    print("RESHAPE NODES")
    print("=" * 120)

    for node in model.graph.node:

        if node.op_type != "Reshape":
            continue

        print("\nName :", node.name)

        print("Inputs")
        for x in node.input:
            print(" ", x, "->", tensor_shapes.get(x))

        print("Outputs")
        for x in node.output:
            print(" ", x, "->", tensor_shapes.get(x))

        if len(node.input) > 1:

            print("Target Shape")

            if node.input[1] in initializer:
                print(initializer[node.input[1]])
            else:
                print("Dynamic :", node.input[1])

    ##############################################################
    # Concat nodes
    ##############################################################

    print("\n")
    print("=" * 120)
    print("CONCAT NODES")
    print("=" * 120)

    for node in model.graph.node:

        if node.op_type != "Concat":
            continue

        print("\nName :", node.name)

        for attr in node.attribute:
            if attr.name == "axis":
                print("Axis :", attr.i)

        print("Inputs")

        for x in node.input:
            print(" ", x, "->", tensor_shapes.get(x))

        print("Outputs")

        for x in node.output:
            print(" ", x, "->", tensor_shapes.get(x))

    ##############################################################
    # MatMul
    ##############################################################

    print("\n")
    print("=" * 120)
    print("MATMUL NODES")
    print("=" * 120)

    for node in model.graph.node:

        if node.op_type != "MatMul":
            continue

        print("\nName :", node.name)

        print("Inputs")

        for x in node.input:
            print(" ", x, "->", tensor_shapes.get(x))

        print("Outputs")

        for x in node.output:
            print(" ", x, "->", tensor_shapes.get(x))

    ##############################################################
    # Graph neighbourhood around Concat
    ##############################################################

    print("\n")
    print("=" * 120)
    print("GRAPH ORDER AROUND CONCAT")
    print("=" * 120)

    nodes = model.graph.node

    for i, node in enumerate(nodes):

        if node.op_type != "Concat":
            continue

        print("\n")
        print("=" * 80)
        print("Concat :", node.name)
        print("Node Index :", i)

        start = max(0, i - 3)
        end = min(len(nodes), i + 4)

        for j in range(start, end):
            n = nodes[j]
            print(f"{j:5d}  {n.op_type:15}  {n.name}")

    ##############################################################
    # ONNX Runtime
    ##############################################################

    print("\n")
    print("=" * 120)
    print("ONNXRUNTIME")
    print("=" * 120)

    try:

        sess = ort.InferenceSession(args.model)

        print("Inputs")

        for i in sess.get_inputs():
            print(i.name)
            print(" Shape :", i.shape)
            print(" Type  :", i.type)
            print()

        print("Outputs")

        for o in sess.get_outputs():
            print(o.name)
            print(" Shape :", o.shape)
            print(" Type  :", o.type)
            print()

    except Exception as e:
        print(e)

    ##############################################################

    print("\n")
    print("=" * 120)
    print("SUMMARY")
    print("=" * 120)

    print("Total Nodes :", len(model.graph.node))
    print("Unique Ops  :", len(cnt))


if __name__ == "__main__":
    main()
