# PicoDet Ball Detection Model

## Supported Models

- model.onnx (ONNX Runtime)
- model.pdmodel, model.pdiparams (Paddle Inference)

## Classes

0 - basketball
1 - volleyball

## Input

- Resolution: 320 × 320
- Color: RGB
- Layout: NCHW

## Preprocessing

- Letterbox resize to 320×320
- Pad value: 114
- Normalize:
  - Mean = [0.485, 0.456, 0.406]
  - Std = [0.229, 0.224, 0.225]

## Output

Each detection is:

[class_id, score, xmin, ymin, xmax, ymax]

## ONNX Runtime

Install:

pip install -r requirements.txt

Run:

python infer_onnx.py \
    --model model.onnx \
    --image sample.jpg \
    --output result.jpg \
    --labels labels.txt
