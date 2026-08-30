import cv2
import numpy as np
import onnxruntime as ort
import argparse
import os


# ============================================================
# Arguments
# ============================================================

parser = argparse.ArgumentParser()

parser.add_argument(
    "--model",
    required=True,
    help="Path to ONNX model"
)

parser.add_argument(
    "--image",
    required=True,
    help="Input image"
)

parser.add_argument(
    "--output",
    default="result.jpg",
    help="Output image"
)

parser.add_argument(
    "--score",
    type=float,
    default=0.5,
    help="Score threshold"
)

parser.add_argument(
    "--labels",
    default=None,
    help="Optional labels.txt"
)

args = parser.parse_args()


# ============================================================
# Load labels
# ============================================================

class_names = None

if args.labels is not None and os.path.exists(args.labels):
    with open(args.labels, "r", encoding="utf-8") as f:
        class_names = [x.strip() for x in f.readlines()]

    print("Loaded labels:", len(class_names))


# ============================================================
# Load ONNX model
# ============================================================

print("\nLoading ONNX model:")
print(args.model)

session = ort.InferenceSession(
    args.model,
    providers=["CPUExecutionProvider"]
)


# ============================================================
# Print ONNX inputs
# ============================================================

print("\n==============================")
print("ONNX INPUTS")
print("==============================")

for inp in session.get_inputs():
    print(
        "Name:",
        inp.name,
        "Shape:",
        inp.shape,
        "Type:",
        inp.type
    )


# ============================================================
# Print ONNX outputs
# ============================================================

print("\n==============================")
print("ONNX OUTPUTS")
print("==============================")

for out in session.get_outputs():
    print(
        "Name:",
        out.name,
        "Shape:",
        out.shape,
        "Type:",
        out.type
    )


# ============================================================
# Get actual ONNX input name
# ============================================================

input_info = session.get_inputs()

if len(input_info) == 0:
    raise RuntimeError("ONNX model has no inputs.")

input_name = input_info[0].name

print("\nUsing ONNX input name:", input_name)


# ============================================================
# Read image
# ============================================================

print("\nReading image:")
print(args.image)

orig = cv2.imread(args.image)

if orig is None:
    raise ValueError(
        "Cannot read image: " + args.image
    )

orig_h, orig_w = orig.shape[:2]

print(
    "Original image size:",
    orig_w,
    "x",
    orig_h
)


# ============================================================
# BGR -> RGB
# ============================================================

img = cv2.cvtColor(
    orig,
    cv2.COLOR_BGR2RGB
)


# ============================================================
# Resize
# ============================================================

input_size = 320

img = cv2.resize(
    img,
    (input_size, input_size)
)


# ============================================================
# Normalize
# ============================================================

img = img.astype(np.float32) / 255.0

mean = np.array(
    [0.485, 0.456, 0.406],
    dtype=np.float32
)

std = np.array(
    [0.229, 0.224, 0.225],
    dtype=np.float32
)

img = (img - mean) / std


# ============================================================
# HWC -> CHW
# ============================================================

img = np.transpose(
    img,
    (2, 0, 1)
)


# ============================================================
# Add batch dimension
# ============================================================

img = np.expand_dims(
    img,
    axis=0
)


# Make absolutely sure it is float32
img = img.astype(np.float32)


print("\n==============================")
print("PREPARED INPUT")
print("==============================")

print("Shape:", img.shape)
print("Dtype:", img.dtype)


# ============================================================
# IMPORTANT:
# Only send the input that the ONNX model actually requires.
#
# Your model says:
#
# images ['batch', 3, 320, 320]
#
# Therefore we DO NOT send:
#
# image
# scale_factor
# ============================================================

feed = {
    input_name: img
}

print("\n==============================")
print("INFERENCE FEED")
print("==============================")

print(feed.keys())

print(
    "Sending:",
    input_name,
    "->",
    img.shape
)


# ============================================================
# Run inference
# ============================================================

outputs = session.run(
    None,
    feed
)


# ============================================================
# Print output information
# ============================================================

print("\n==============================")
print("RAW OUTPUTS")
print("==============================")

for i, output in enumerate(outputs):

    print(
        "Output",
        i,
        "shape:",
        np.asarray(output).shape,
        "dtype:",
        np.asarray(output).dtype
    )


# ============================================================
# Your model appears to have:
#
# Output 0:
# [batch, 2125, 84]
#
# Output 1:
# number of boxes
#
# ============================================================

if len(outputs) == 0:
    raise RuntimeError(
        "ONNX model returned no outputs."
    )


boxes = outputs[0]

print("\nOutput[0] shape:", boxes.shape)


# ============================================================
# Number of detections
# ============================================================

num_boxes = None

if len(outputs) > 1:

    try:
        num_boxes = int(
            np.asarray(outputs[1]).reshape(-1)[0]
        )

    except Exception:
        num_boxes = None


if num_boxes is not None:

    print(
        "\nDetections reported by model:",
        num_boxes
    )


# ============================================================
# Remove batch dimension
# ============================================================

pred = np.asarray(boxes)

if pred.ndim == 3:
    pred = pred[0]

print(
    "Prediction shape after batch removal:",
    pred.shape
)


# ============================================================
# Save raw output for debugging
# ============================================================

np.save(
    "onnx_output.npy",
    pred
)

print(
    "\nRaw output saved to:",
    os.path.abspath("onnx_output.npy")
)


# ============================================================
# Detection processing
#
# Expected format for 84 values:
#
# [x1, y1, x2, y2, class_score_0, ... class_score_79]
#
# 4 + 80 = 84
#
# ============================================================

detections = []


if pred.ndim == 2 and pred.shape[1] == 84:

    print(
        "\nDetected 84-value output format."
    )

    # If model reports a number of boxes,
    # don't process more than that.
    if num_boxes is not None:
        count = min(
            num_boxes,
            pred.shape[0]
        )
    else:
        count = pred.shape[0]

    pred = pred[:count]

    # First 4 values = bounding box
    box_data = pred[:, :4]

    # Remaining 80 values = class scores
    class_scores = pred[:, 4:]

    # Best class
    class_ids = np.argmax(
        class_scores,
        axis=1
    )

    scores = np.max(
        class_scores,
        axis=1
    )

    # ========================================================
    # Process detections
    # ========================================================

    for i in range(len(pred)):

        score = float(scores[i])

        if score < args.score:
            continue

        class_id = int(class_ids[i])

        x1, y1, x2, y2 = box_data[i]

        # ----------------------------------------------------
        # Determine whether coordinates are normalized
        # or already in 320x320 coordinates.
        # ----------------------------------------------------

        if (
            abs(x1) <= 2
            and abs(y1) <= 2
            and abs(x2) <= 2
            and abs(y2) <= 2
        ):
            # Normalized coordinates
            x1 *= input_size
            y1 *= input_size
            x2 *= input_size
            y2 *= input_size

        # ----------------------------------------------------
        # Scale from 320x320 back to original image
        # ----------------------------------------------------

        x1 = x1 * orig_w / input_size
        x2 = x2 * orig_w / input_size

        y1 = y1 * orig_h / input_size
        y2 = y2 * orig_h / input_size

        # Clamp
        x1 = max(0, min(orig_w - 1, x1))
        x2 = max(0, min(orig_w - 1, x2))

        y1 = max(0, min(orig_h - 1, y1))
        y2 = max(0, min(orig_h - 1, y2))

        detections.append(
            (
                int(x1),
                int(y1),
                int(x2),
                int(y2),
                score,
                class_id
            )
        )


# ============================================================
# Alternative output format
# ============================================================

elif pred.ndim == 2 and pred.shape[1] >= 6:

    print(
        "\nUsing generic detection output format."
    )

    if num_boxes is not None:
        count = min(
            num_boxes,
            pred.shape[0]
        )
    else:
        count = pred.shape[0]

    pred = pred[:count]

    for row in pred:

        x1 = float(row[0])
        y1 = float(row[1])
        x2 = float(row[2])
        y2 = float(row[3])

        score = float(row[4])
        class_id = int(row[5])

        if score < args.score:
            continue

        # Coordinates in 320x320
        x1 = x1 * orig_w / input_size
        x2 = x2 * orig_w / input_size

        y1 = y1 * orig_h / input_size
        y2 = y2 * orig_h / input_size

        x1 = max(0, min(orig_w - 1, x1))
        x2 = max(0, min(orig_w - 1, x2))

        y1 = max(0, min(orig_h - 1, y1))
        y2 = max(0, min(orig_h - 1, y2))

        detections.append(
            (
                int(x1),
                int(y1),
                int(x2),
                int(y2),
                score,
                class_id
            )
        )


else:

    print(
        "\nWARNING:"
    )

    print(
        "Unknown output format:",
        pred.shape
    )

    print(
        "The ONNX inference itself completed successfully."
    )


# ============================================================
# Draw detections
# ============================================================

print("\n==============================")
print("DETECTIONS")
print("==============================")


for i, detection in enumerate(detections):

    x1, y1, x2, y2, score, class_id = detection

    # Label
    if (
        class_names is not None
        and class_id < len(class_names)
    ):
        label_name = class_names[class_id]

    else:
        label_name = f"class_{class_id}"

    label = (
        f"{label_name}: "
        f"{score:.2f}"
    )

    print(
        i,
        label,
        f"box=({x1},{y1},{x2},{y2})"
    )

    # Draw rectangle
    cv2.rectangle(
        orig,
        (x1, y1),
        (x2, y2),
        (0, 255, 0),
        2
    )

    # Text position
    text_y = max(
        20,
        y1 - 5
    )

    cv2.putText(
        orig,
        label,
        (x1, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2,
        cv2.LINE_AA
    )


# ============================================================
# Print final result
# ============================================================

print(
    "\nFinal detections above threshold:",
    len(detections)
)


# ============================================================
# Save result
# ============================================================

cv2.imwrite(
    args.output,
    orig
)

print(
    "\nResult saved to:"
)

print(
    os.path.abspath(args.output)
)

print("\nDone.")
