import onnx
from collections import Counter, defaultdict
import os
import sys


# ============================================================
# CHANGE ONLY THIS
# ============================================================
MODEL = r"picodet_new.onnx"
# Example:
# MODEL = r"output_infer/picodet_leakyrelu.onnx"


# ============================================================
# LOAD MODEL
# ============================================================
print("=" * 70)
print("NIKON ONNX COMPATIBILITY CHECK")
print("=" * 70)

print("\nMODEL :", MODEL)

if not os.path.exists(MODEL):
    print("\nERROR: Model file not found.")
    print("Check the MODEL path.")
    sys.exit(1)

try:
    model = onnx.load(MODEL)
    print("Load : SUCCESS")
except Exception as e:
    print("Load : FAILED")
    print(e)
    sys.exit(1)


# ============================================================
# ONNX CHECKER
# ============================================================
print("\n" + "=" * 70)
print("ONNX CHECKER")
print("=" * 70)

try:
    onnx.checker.check_model(model)
    print("ONNX checker : PASS")
except Exception as e:
    print("ONNX checker : FAIL")
    print(e)


# ============================================================
# OPERATOR COUNTS
# ============================================================
print("\n" + "=" * 70)
print("OPERATOR COUNTS")
print("=" * 70)

ops = Counter(node.op_type for node in model.graph.node)

for op, count in sorted(ops.items()):
    print(f"{op:25s} {count}")


# ============================================================
# INPUTS
# ============================================================
print("\n" + "=" * 70)
print("MODEL INPUTS")
print("=" * 70)

for inp in model.graph.input:

    shape = []

    for d in inp.type.tensor_type.shape.dim:

        if d.HasField("dim_value"):
            shape.append(str(d.dim_value))

        elif d.HasField("dim_param"):
            shape.append(d.dim_param)

        else:
            shape.append("?")

    print("\nName :", inp.name)
    print("Shape:", shape)
    print("Type :", inp.type.tensor_type.elem_type)


# ============================================================
# OUTPUTS
# ============================================================
print("\n" + "=" * 70)
print("MODEL OUTPUTS")
print("=" * 70)

for out in model.graph.output:

    shape = []

    for d in out.type.tensor_type.shape.dim:

        if d.HasField("dim_value"):
            shape.append(str(d.dim_value))

        elif d.HasField("dim_param"):
            shape.append(d.dim_param)

        else:
            shape.append("?")

    print("\nName :", out.name)
    print("Shape:", shape)


# ============================================================
# DYNAMIC DIMENSION CHECK
# ============================================================
print("\n" + "=" * 70)
print("DYNAMIC DIMENSION CHECK")
print("=" * 70)

dynamic_dims = []

for inp in model.graph.input:

    for i, d in enumerate(inp.type.tensor_type.shape.dim):

        if d.HasField("dim_param"):
            dynamic_dims.append(
                ("INPUT", inp.name, i, d.dim_param)
            )

        elif not d.HasField("dim_value"):
            dynamic_dims.append(
                ("INPUT", inp.name, i, "?")
            )

for out in model.graph.output:

    for i, d in enumerate(out.type.tensor_type.shape.dim):

        if d.HasField("dim_param"):
            dynamic_dims.append(
                ("OUTPUT", out.name, i, d.dim_param)
            )

        elif not d.HasField("dim_value"):
            dynamic_dims.append(
                ("OUTPUT", out.name, i, "?")
            )

if dynamic_dims:

    print("WARNING: Dynamic dimensions found.")

    for x in dynamic_dims:
        print(
            f"{x[0]:7s} name={x[1]}  "
            f"dimension={x[2]}  value={x[3]}"
        )

else:
    print("No dynamic input/output dimensions found.")


# ============================================================
# RESHAPE NODES
# ============================================================
print("\n" + "=" * 70)
print("RESHAPE NODES")
print("=" * 70)

reshape_nodes = [
    node for node in model.graph.node
    if node.op_type == "Reshape"
]

print("Total Reshape nodes :", len(reshape_nodes))

for node in reshape_nodes:

    print("\nName :", node.name)
    print("Inputs:")

    for x in node.input:
        print(" ", x)

    print("Outputs:")

    for x in node.output:
        print(" ", x)


# ============================================================
# CONCAT NODES
# ============================================================
print("\n" + "=" * 70)
print("CONCAT NODES")
print("=" * 70)

concat_nodes = [
    node for node in model.graph.node
    if node.op_type == "Concat"
]

print("Total Concat nodes :", len(concat_nodes))

for node in concat_nodes:

    axis = "unknown"

    for attr in node.attribute:
        if attr.name == "axis":
            axis = attr.i

    print("\nName :", node.name)
    print("Axis :", axis)

    print("Inputs:")
    for x in node.input:
        print(" ", x)

    print("Outputs:")
    for x in node.output:
        print(" ", x)


# ============================================================
# MATMUL NODES
# ============================================================
print("\n" + "=" * 70)
print("MATMUL NODES")
print("=" * 70)

matmul_nodes = [
    node for node in model.graph.node
    if node.op_type == "MatMul"
]

print("Total MatMul nodes :", len(matmul_nodes))

for node in matmul_nodes:

    print("\nName :", node.name)

    print("Inputs:")
    for x in node.input:
        print(" ", x)

    print("Outputs:")
    for x in node.output:
        print(" ", x)


# ============================================================
# GATHER NODES
# ============================================================
print("\n" + "=" * 70)
print("GATHER NODES")
print("=" * 70)

gather_nodes = [
    node for node in model.graph.node
    if node.op_type == "Gather"
]

print("Total Gather nodes :", len(gather_nodes))

for node in gather_nodes:

    axis = 0

    for attr in node.attribute:
        if attr.name == "axis":
            axis = attr.i

    print("\nName :", node.name)
    print("Axis :", axis)

    print("Inputs:")
    for x in node.input:
        print(" ", x)

    print("Outputs:")
    for x in node.output:
        print(" ", x)


# ============================================================
# RESHAPE -> CONCAT RELATIONSHIP
# ============================================================
print("\n" + "=" * 70)
print("RESHAPE -> CONCAT RELATIONSHIP")
print("=" * 70)

reshape_outputs = {}

for node in reshape_nodes:
    for output in node.output:
        reshape_outputs[output] = node.name


for node in concat_nodes:

    related = []

    for inp in node.input:

        if inp in reshape_outputs:
            related.append(
                f"{reshape_outputs[inp]} -> {inp}"
            )

    if related:

        print("\nConcat :", node.name)

        for x in related:
            print(" ", x)


# ============================================================
# MATMUL INPUT SOURCES
# ============================================================
print("\n" + "=" * 70)
print("MATMUL INPUT SOURCES")
print("=" * 70)

producer = {}

for node in model.graph.node:
    for output in node.output:
        producer[output] = node


for node in matmul_nodes:

    print("\nMatMul :", node.name)

    for inp in node.input:

        if inp in producer:
            src = producer[inp]
            print(
                f"  {inp} <- {src.op_type} ({src.name})"
            )
        else:
            print(
                f"  {inp} <- graph input/initializer"
            )


# ============================================================
# FINAL ASSESSMENT
# ============================================================
print("\n" + "=" * 70)
print("FINAL ASSESSMENT")
print("=" * 70)

issues = []

if dynamic_dims:
    issues.append(
        "Dynamic dimensions are present."
    )

if len(gather_nodes) > 0:
    issues.append(
        f"{len(gather_nodes)} Gather node(s) are present."
    )

if len(reshape_nodes) > 0:
    issues.append(
        f"{len(reshape_nodes)} Reshape node(s) are present."
    )

if len(concat_nodes) > 0:
    issues.append(
        f"{len(concat_nodes)} Concat node(s) are present."
    )

if len(matmul_nodes) > 0:
    issues.append(
        f"{len(matmul_nodes)} MatMul node(s) are present."
    )


if issues:

    print("\nItems requiring Nikon/compiler attention:")

    for issue in issues:
        print(" -", issue)

    print(
        "\nThis does NOT mean the model is wrong."
    )

    print(
        "It means the graph contains the same types of "
        "operators/shapes involved in the previous compiler issue."
    )

else:

    print("\nNo obvious dynamic-shape / MatMul / Gather / "
          "Reshape / Concat issue found.")

print("\n" + "=" * 70)
print("CHECK COMPLETE")
print("=" * 70)
