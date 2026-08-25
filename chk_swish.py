import sys
import onnx

if len(sys.argv) != 2:
    print("Usage: python check_swish.py model.onnx")
    sys.exit(1)

model = onnx.load(sys.argv[1])

nodes = list(model.graph.node)

# Map each tensor output -> node that produces it
producer = {}

for node in nodes:
    for output in node.output:
        producer[output] = node


print("=" * 100)
print("SWISH / HARD-SWISH PATTERN CHECK")
print("=" * 100)

found_sigmoid = 0
found_hardsigmoid = 0

for node in nodes:

    if node.op_type != "Mul":
        continue

    if len(node.input) != 2:
        continue

    a = producer.get(node.input[0])
    b = producer.get(node.input[1])

    # Check Sigmoid -> Mul
    if a and a.op_type == "Sigmoid":
        found_sigmoid += 1

        print("\n[Sigmoid -> Mul]")
        print("Sigmoid :", a.name)
        print("Output  :", a.output[0])
        print("Mul     :", node.name)
        print("Inputs  :", list(node.input))

    elif b and b.op_type == "Sigmoid":
        found_sigmoid += 1

        print("\n[Sigmoid -> Mul]")
        print("Sigmoid :", b.name)
        print("Output  :", b.output[0])
        print("Mul     :", node.name)
        print("Inputs  :", list(node.input))

    # Check HardSigmoid -> Mul
    if a and a.op_type == "HardSigmoid":
        found_hardsigmoid += 1

        print("\n[HardSigmoid -> Mul]")
        print("HardSigmoid :", a.name)
        print("Output      :", a.output[0])
        print("Mul         :", node.name)
        print("Inputs      :", list(node.input))

    elif b and b.op_type == "HardSigmoid":
        found_hardsigmoid += 1

        print("\n[HardSigmoid -> Mul]")
        print("HardSigmoid :", b.name)
        print("Output      :", b.output[0])
        print("Mul         :", node.name)
        print("Inputs      :", list(node.input))


print("\n" + "=" * 100)
print("SUMMARY")
print("=" * 100)

print("Sigmoid -> Mul pairs     :", found_sigmoid)
print("HardSigmoid -> Mul pairs :", found_hardsigmoid)

print("\nInterpretation:")
print("  Sigmoid -> Mul      = possible Swish")
print("  HardSigmoid -> Mul  = possible HardSwish")
print("=" * 100)
