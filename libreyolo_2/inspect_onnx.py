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


def get_attr(node, name, default=None):
    for attr in node.attribute:
        if attr.name == name:
            if attr.type == onnx.AttributeProto.INT:
                return attr.i
            if attr.type == onnx.AttributeProto.INTS:
                return list(attr.ints)
    return default


def main():

    parser = argparse.ArgumentParser(
        description="Inspect ONNX model for compiler compatibility"
    )

    parser.add_argument(
        "model",
        help="Path to ONNX model"
    )

    args = parser.parse_args()

    print("=" * 120)
    print("MODEL")
    print("=" * 120)
    print(args.model)

    model = onnx.load(args.model)

    # ------------------------------------------------------------
    # Shape inference
    # ------------------------------------------------------------

    try:
        model = shape_inference.infer_shapes(model)
        print("\nShape inference: SUCCESS")
    except Exception as e:
        print("\nShape inference: FAILED")
        print(type(e).__name__, ":", e)

    # ------------------------------------------------------------
    # Tensor shape dictionary
    # ------------------------------------------------------------

    tensor_shapes = {}

    all_values = (
        list(model.graph.input)
        + list(model.graph.value_info)
        + list(model.graph.output)
    )

    for value in all_values:
        tensor_shapes[value.name] = get_shape(value)

    # ------------------------------------------------------------
    # Initializers
    # ------------------------------------------------------------

    initializers = {}

    for tensor in model.graph.initializer:
        initializers[tensor.name] = numpy_helper.to_array(tensor)

    # ------------------------------------------------------------
    # INPUTS
    # ------------------------------------------------------------

    print("\n" + "=" * 120)
    print("MODEL INPUTS")
    print("=" * 120)

    for inp in model.graph.input:

        print("\nName :", inp.name)
        print("Shape:", get_shape(inp))
        print(
            "DType:",
            dtype_name(inp.type.tensor_type.elem_type)
        )

    # ------------------------------------------------------------
    # OUTPUTS
    # ------------------------------------------------------------

    print("\n" + "=" * 120)
    print("MODEL OUTPUTS")
    print("=" * 120)

    for out in model.graph.output:

        print("\nName :", out.name)
        print("Shape:", get_shape(out))
        print(
            "DType:",
            dtype_name(out.type.tensor_type.elem_type)
        )

    # ------------------------------------------------------------
    # OPERATOR COUNTS
    # ------------------------------------------------------------

    print("\n" + "=" * 120)
    print("OPERATOR COUNTS")
    print("=" * 120)

    counts = Counter(
        node.op_type
        for node in model.graph.node
    )

    for op in sorted(counts):
        print(f"{op:25} {counts[op]}")

    # ------------------------------------------------------------
    # RESHAPE NODES
    # ------------------------------------------------------------

    print("\n" + "=" * 120)
    print("RESHAPE NODES")
    print("=" * 120)

    for node in model.graph.node:

        if node.op_type != "Reshape":
            continue

        print("\nName:", node.name)

        print("Inputs:")

        for x in node.input:
            print(
                "  ",
                x,
                "->",
                tensor_shapes.get(x)
            )

        print("Outputs:")

        for x in node.output:
            print(
                "  ",
                x,
                "->",
                tensor_shapes.get(x)
            )

        if len(node.input) > 1:

            shape_input = node.input[1]

            if shape_input in initializers:

                print(
                    "Target shape:",
                    initializers[shape_input]
                )

            else:

                print(
                    "Target shape:",
                    "DYNAMIC",
                    shape_input
                )

    # ------------------------------------------------------------
    # CONCAT NODES
    # ------------------------------------------------------------

    print("\n" + "=" * 120)
    print("CONCAT NODES")
    print("=" * 120)

    for node in model.graph.node:

        if node.op_type != "Concat":
            continue

        print("\nName:", node.name)

        print(
            "Axis:",
            get_attr(node, "axis")
        )

        print("Inputs:")

        for x in node.input:

            print(
                "  ",
                x,
                "->",
                tensor_shapes.get(x)
            )

        print("Outputs:")

        for x in node.output:

            print(
                "  ",
                x,
                "->",
                tensor_shapes.get(x)
            )

    # ------------------------------------------------------------
    # MATMUL NODES
    # ------------------------------------------------------------

    print("\n" + "=" * 120)
    print("MATMUL NODES")
    print("=" * 120)

    for node in model.graph.node:

        if node.op_type != "MatMul":
            continue

        print("\nName:", node.name)

        print("Inputs:")

        for x in node.input:

            print(
                "  ",
                x,
                "->",
                tensor_shapes.get(x)
            )

        print("Outputs:")

        for x in node.output:

            print(
                "  ",
                x,
                "->",
                tensor_shapes.get(x)
            )

    # ------------------------------------------------------------
    # GATHER NODES
    # ------------------------------------------------------------

    print("\n" + "=" * 120)
    print("GATHER NODES")
    print("=" * 120)

    for node in model.graph.node:

        if node.op_type != "Gather":
            continue

        print("\nName:", node.name)

        print("Inputs:")

        for x in node.input:

            print(
                "  ",
                x,
                "->",
                tensor_shapes.get(x)
            )

        print("Outputs:")

        for x in node.output:

            print(
                "  ",
                x,
                "->",
                tensor_shapes.get(x)
            )

        print(
            "Axis:",
            get_attr(node, "axis")
        )

    # ------------------------------------------------------------
    # SHAPE NODES
    # ------------------------------------------------------------

    print("\n" + "=" * 120)
    print("SHAPE NODES")
    print("=" * 120)

    for node in model.graph.node:

        if node.op_type != "Shape":
            continue

        print("\nName:", node.name)

        for x in node.input:

            print(
                "Input :",
                x,
                "->",
                tensor_shapes.get(x)
            )

        for x in node.output:

            print(
                "Output:",
                x,
                "->",
                tensor_shapes.get(x)
            )

    # ------------------------------------------------------------
    # GRAPH ORDER AROUND CONCAT
    # ------------------------------------------------------------

    print("\n" + "=" * 120)
    print("GRAPH ORDER AROUND CONCAT")
    print("=" * 120)

    nodes = model.graph.node

    for i, node in enumerate(nodes):

        if node.op_type != "Concat":
            continue

        print("\n" + "-" * 80)

        print(
            "Concat:",
            node.name
        )

        print(
            "Node index:",
            i
        )

        start = max(0, i - 5)
        end = min(len(nodes), i + 6)

        for j in range(start, end):

            n = nodes[j]

            print(
                f"{j:5d}  "
                f"{n.op_type:15}  "
                f"{n.name}"
            )

    # ------------------------------------------------------------
    # ONNX RUNTIME
    # ------------------------------------------------------------

    print("\n" + "=" * 120)
    print("ONNX RUNTIME")
    print("=" * 120)

    try:

        session = ort.InferenceSession(
            args.model
        )

        print("\nInputs:")

        for inp in session.get_inputs():

            print(
                inp.name,
                "shape=",
                inp.shape,
                "type=",
                inp.type
            )

        print("\nOutputs:")

        for out in session.get_outputs():

            print(
                out.name,
                "shape=",
                out.shape,
                "type=",
                out.type
            )

    except Exception as e:

        print(
            "ONNX Runtime inspection failed:",
            type(e).__name__,
            ":",
            e
        )

    # ------------------------------------------------------------
    # SUMMARY
    # ------------------------------------------------------------

    print("\n" + "=" * 120)
    print("SUMMARY")
    print("=" * 120)

    print(
        "Total nodes :",
        len(model.graph.node)
    )

    print(
        "Unique ops  :",
        len(counts)
    )


if __name__ == "__main__":
    main()
