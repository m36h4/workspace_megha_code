"""
Verify an exported PicoDet ONNX model at 320x480:
1. Loads correctly in ONNX Runtime
2. Reports the exact input/output names and shapes it expects
3. Runs a real forward pass with correctly-preprocessed dummy/real data
4. Confirms output shape is sane

Usage:
  python3 verify_onnx_320x480.py path/to/last.onnx [path/to/real_image.jpg]
"""
import sys

import numpy as np
import onnxruntime as ort

IMAGENET_MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
IMAGENET_STD = np.array([58.395, 57.12, 57.375], dtype=np.float32)


def preprocess_letterbox(img_bgr, target_h=320, target_w=480, pad_value=114):
    import cv2
    orig_h, orig_w = img_bgr.shape[:2]
    ratio = min(target_h / orig_h, target_w / orig_w)
    new_h, new_w = int(round(orig_h * ratio)), int(round(orig_w * ratio))

    rgb = img_bgr[:, :, ::-1]
    resized = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    canvas = np.full((target_h, target_w, 3), pad_value, dtype=np.uint8)
    canvas[:new_h, :new_w] = resized  # top-left anchor, matches training

    chw = canvas.astype(np.float32)
    chw = (chw - IMAGENET_MEAN) / IMAGENET_STD
    chw = chw.transpose(2, 0, 1)
    return chw[np.newaxis, ...].astype(np.float32), ratio  # (1, 3, H, W)


def main():
    onnx_path = sys.argv[1]
    img_path = sys.argv[2] if len(sys.argv) > 2 else None

    print(f"Loading {onnx_path} ...")
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

    print("\n=== Model inputs ===")
    for inp in session.get_inputs():
        print(f"  name={inp.name!r}  shape={inp.shape}  dtype={inp.type}")

    print("\n=== Model outputs ===")
    for out in session.get_outputs():
        print(f"  name={out.name!r}  shape={out.shape}  dtype={out.type}")

    input_name = session.get_inputs()[0].name
    expected_shape = session.get_inputs()[0].shape  # e.g. [1, 3, 320, 480] or with dynamic dims

    # Build a real or dummy input tensor at exactly 320x480
    if img_path:
        import cv2
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            raise SystemExit(f"Could not read image at {img_path}")
        input_tensor, ratio = preprocess_letterbox(img_bgr, target_h=320, target_w=480)
        print(f"\nUsing real image {img_path}, letterbox ratio={ratio:.4f}")
    else:
        # Dummy input -- still proves the graph runs at this exact shape
        input_tensor = np.random.randn(1, 3, 320, 480).astype(np.float32)
        print("\nNo image path given -- using random dummy input for a pure shape/graph-execution test")

    print(f"Feeding input of shape {input_tensor.shape} into input {input_name!r} ...")
    outputs = session.run(None, {input_name: input_tensor})

    print("\n=== Inference ran successfully ===")
    for i, out in enumerate(outputs):
        print(f"  output[{i}] shape={out.shape} dtype={out.dtype}")
        print(f"    min={out.min():.4f} max={out.max():.4f} mean={out.mean():.4f}")


if __name__ == "__main__":
    main()
