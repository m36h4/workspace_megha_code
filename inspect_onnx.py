import onnx
from collections import Counter

MODEL = "Pytorch_picodet_upd_inputsz/pytorch_picodet_sim.onnx"

model = onnx.load(MODEL)
graph = model.graph

print("=" * 60)
print("MODEL INPUTS")
print("=" * 60)

for x in graph.input:
    print(x.name, x.type)

print("\n" + "=" * 60)
print("MODEL OUTPUTS")
print("=" * 60)

for x in graph.output:
    print(x.name, x.type)

print("\n" + "=" * 60)
print("OPERATOR SUMMARY")
print("=" * 60)

ops = Counter(node.op_type for node in graph.node)

for op, count in sorted(ops.items()):
    print(f"{op:30} {count}")

print("\n" + "=" * 60)
print("NMS CHECK")
print("=" * 60)

nms_ops = [
    node for node in graph.node
    if "NonMaxSuppression" in node.op_type
    or "NMS" in node.op_type.upper()
]

if nms_ops:
    print("NMS FOUND:")
    for node in nms_ops:
        print(node.op_type, node.name)
else:
    print("NO NMS OPERATOR FOUND IN ONNX GRAPH")

print("\n" + "=" * 60)
print("CLASSIFICATION / POSTPROCESSING OPS")
print("=" * 60)

interesting = [
    "Sigmoid",
    "Softmax",
    "ArgMax",
    "TopK",
    "NonMaxSuppression",
    "Concat",
    "ReduceMax",
    "ReduceSum"
]

for node in graph.node:
    if node.op_type in interesting:
        print(f"{node.op_type:25} {node.name}")
        print(f"    inputs : {list(node.input)}")
        print(f"    outputs: {list(node.output)}")

print("\n" + "=" * 60)
print("FINAL OUTPUT PRODUCER")
print("=" * 60)

output_names = {x.name for x in graph.output}

for node in graph.node:
    for out in node.output:
        if out in output_names:
            print("Output:", out)
            print("Produced by:", node.op_type)
            print("Node name:", node.name)
            print("Inputs:", list(node.input))
