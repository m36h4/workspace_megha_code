import cv2
import numpy as np
import onnxruntime as ort
import argparse
import os

# ----------------------------
# Arguments
# ----------------------------
parser = argparse.ArgumentParser()
parser.add_argument(
    "--model",
    default="model.onnx",
    help="Path to ONNX model",
)
parser.add_argument(
    "--image",
    required=True,
    help="Input image",
)
parser.add_argument(
    "--output",
    default="result.jpg",
    help="Output image",
)
parser.add_argument(
    "--score",
    type=float,
    default=0.5,
    help="Score threshold",
)
parser.add_argument(
    "--labels",
    default=None,
    help="labels.txt (optional)",
)

args = parser.parse_args()

# ----------------------------
# Load labels
# ----------------------------
class_names = None

if args.labels is not None and os.path.exists(args.labels):
    with open(args.labels, "r") as f:
        class_names = [x.strip() for x in f.readlines()]

# ----------------------------
# Load ONNX model
# ----------------------------
session = ort.InferenceSession(
    args.model,
    providers=["CPUExecutionProvider"]
)

print("Inputs:")
for inp in session.get_inputs():
    print(inp.name, inp.shape)

print("\nOutputs:")
for out in session.get_outputs():
    print(out.name, out.shape)

# ----------------------------
# Read image
# ----------------------------
orig = cv2.imread(args.image)

if orig is None:
    raise ValueError("Cannot read image")

h, w = orig.shape[:2]

img = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)

# ----------------------------
# Resize
# ----------------------------
input_size = 320

img = cv2.resize(img, (input_size, input_size))

# ----------------------------
# Normalize
# ----------------------------
img = img.astype(np.float32) / 255.0

mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

img = (img - mean) / std

# HWC -> CHW
img = np.transpose(img, (2, 0, 1))

# Batch dimension
img = np.expand_dims(img, axis=0)

# Scale factor
scale_factor = np.array(
    [[input_size / h, input_size / w]],
    dtype=np.float32,
)

# ----------------------------
# Inference
# ----------------------------
outputs = session.run(
    None,
    {
        "image": img,
        "scale_factor": scale_factor,
    },
)

boxes = outputs[0]
num_boxes = int(outputs[1][0])

print(f"\nDetections: {num_boxes}")

# ----------------------------
# Draw boxes
# ----------------------------
for det in boxes:

    cls = int(det[0])
    score = float(det[1])

    if score < args.score:
        continue

    x1 = int(det[2])
    y1 = int(det[3])
    x2 = int(det[4])
    y2 = int(det[5])

    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(0, min(x2, w - 1))
    y2 = max(0, min(y2, h - 1))

    cv2.rectangle(
        orig,
        (x1, y1),
        (x2, y2),
        (0, 255, 0),
        2,
    )

    if class_names is not None and cls < len(class_names):
        label = f"{class_names[cls]} {score:.2f}"
    else:
        label = f"Class {cls} {score:.2f}"

    cv2.putText(
        orig,
        label,
        (x1, max(20, y1 - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2,
    )

# ----------------------------
# Save result
# ----------------------------
cv2.imwrite(args.output, orig)

print(f"\nSaved: {args.output}")
