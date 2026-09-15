# Ball Detection — Evaluation & Inference Toolkit

Implements the evaluation spec from the client slides:
- **Metric**: Recall @ FPPI=0.05, IoU > 0.4 (single class: "ball")
- **Sports-wise recall** breakdown (basketball / soccer / volleyball)
- **GMACs** via torchinfo (PyTorch models only)
- **Loss curves** and **FP/FN sample dumps** for selected models
- Ground truth **annotation format**: reads **three** formats — client (Nikon
  JSON), COCO, and YOLO — converting all of them internally to the same
  `[x, y, w, h]` top-left-based, original-image-pixel box convention before
  scoring. Verified byte-identical results across all three formats on a
  shared synthetic test set.

## Scripts (3 model backends × dedicated eval + infer script each)

| Model format | Eval script | Infer script | Status |
|---|---|---|---|
| ONNX | `scripts/eval_onnx.py` | `scripts/infer_onnx.py` | ✅ Working, tested end-to-end (dummy model) |
| NanoDet-Plus `.pt` | `scripts/eval_nanodet.py` | `scripts/infer_nanodet.py` | ⏳ CLI/pipeline done; needs `models/nanodet_backend.py` completed (see below) |
| PicoDet `.pdparams` | `scripts/eval_picodet.py` | `scripts/infer_picodet.py` | ⏳ CLI/pipeline done; needs `models/picodet_backend.py` completed (see below) |

Every eval/infer script accepts the **same three annotation-format flags**:

```
--ann-format client --data-root <dir>                                   # Nikon format
--ann-format coco   --coco-ann <instances.json> --coco-image-dir <dir>  # COCO format
--ann-format yolo   --yolo-images <dir> --yolo-labels <dir>             # YOLO format
```

## Status

| Component | Status |
|---|---|
| Dataset loaders — client / coco / yolo | ✅ Done, tested (all 3 converge to identical GT boxes on shared fixtures) |
| Preprocessing (letterbox resize, no aspect distortion) | ✅ Done, tested |
| IoU matching + Recall@FPPI metric engine | ✅ Done, tested against synthetic ground-truth cases |
| Report generation (markdown + JSON + plots + FP/FN overlays) | ✅ Done, tested |
| **ONNX inference backend** | ✅ Done, tested with a dummy model across all 3 GT formats — **still needs validation against your real exported model's actual output format** (see below) |
| **NanoDet-Plus `.pt` backend** | ⏳ Scaffolded; waiting on checkpoint + training config |
| **PicoDet `.pdparams` backend** | ⏳ Scaffolded; waiting on checkpoint + config, and a decision on export path (see below) |

## What I still need from you

1. **NanoDet-Plus**: the `.pt` checkpoint **and** the exact model config (`.yml`) used to train it (backbone, head type, `reg_max`, strides, input size, class count). NanoDet's raw output is undecoded per-stride logits — without the config I can't build the correct decode function.
2. **PicoDet**: the `.pdparams` **and** its PaddleDetection config `.yml`. Important — since we're avoiding a `paddledet` runtime dependency for the client deliverable, the practical path is:
   - **Option A (recommended)**: export PicoDet to ONNX once (I'll write this one-time export script using `paddledet`, since Paddle's tools are needed regardless to interpret the `.pdparams` architecture) → then it runs through the same ONNX backend as everything else, no Paddle needed at inference time.
   - **Option B**: export to Paddle Inference format (`.pdmodel` + `.pdiparams`) and use `paddle.inference` (lighter than full `paddledet`, but still a Paddle runtime dependency on the client machine).
   Let me know which you'd prefer — Option A is cleanest for the client since it unifies everything under one ONNX runner.
3. **Confirm ONNX output convention** for your actual exported model: is post-processing (NMS + box decoding) baked into the graph, or does it output raw per-stride head tensors? This determines whether `models/onnx_backend.py::_parse_output()` works as-is or needs the architecture-specific decode logic.
4. **Confirm normalization**: 0–1 scaling or ImageNet mean/std? RGB or BGR? (defaults are 0–1 scaling, RGB — override via `--mean`/`--std` flags either way).
5. A couple of real sample images + label JSONs (or point me at the dataset path) to run a real correctness check, not just synthetic.

## Usage (ONNX — works today)

```bash
# Evaluation — client (Nikon) format
python scripts/eval_onnx.py \
    --onnx model.onnx \
    --ann-format client --data-root /path/to/basic_eval_caseA \
    --input-w 480 --input-h 320 \
    --mean 0 0 0 --std 255 255 255 \
    --iou-thresh 0.4 --target-fppi 0.05 \
    --out-dir results/onnx_eval \
    --dump-fp-fn

# Evaluation — COCO format
python scripts/eval_onnx.py \
    --onnx model.onnx \
    --ann-format coco --coco-ann instances.json --coco-image-dir /path/to/images \
    --input-w 480 --input-h 320 --out-dir results/onnx_eval_coco

# Evaluation — YOLO format
python scripts/eval_onnx.py \
    --onnx model.onnx \
    --ann-format yolo --yolo-images /path/to/images --yolo-labels /path/to/labels \
    --input-w 480 --input-h 320 --out-dir results/onnx_eval_yolo

# Visual sanity-check inference (no labels needed; optional --ann-format overlays GT)
python scripts/infer_onnx.py \
    --onnx model.onnx \
    --input /path/to/images_or_single_image.jpg \
    --out-dir vis_output \
    --score-thresh 0.3
```

`eval_nanodet.py` / `eval_picodet.py` and their `infer_*` counterparts accept
the identical `--ann-format` flags — same usage pattern, once their model
backends are completed (see Status table).

## Project layout

```
core/
    dataset.py       — 3 GT loaders: client (Nikon JSON), COCO, YOLO — all converge to
                       the same [x,y,w,h] top-left, original-pixel box convention
    dataset_cli.py   — shared argparse wiring so every script exposes the same
                       --ann-format client|coco|yolo options consistently
    preprocess.py    — letterbox resize (no aspect distortion, per spec), box format conversions
    metrics.py       — IoU matching, Recall@FPPI sweep, per-sport breakdown, FP/FN collection
    nms.py           — class-agnostic NMS (single class = ball)
    report.py        — markdown/JSON report, GMACs via torchinfo, loss curve + FP/FN plotting
models/
    onnx_backend.py       — ONNX Runtime inference (done)
    nanodet_backend.py    — PyTorch .pt inference (scaffolded; pending your config/weights)
    picodet_backend.py    — Paddle inference (scaffolded; pending your config/weights + export decision)
scripts/
    eval_onnx.py     / infer_onnx.py      — ONNX (working)
    eval_nanodet.py  / infer_nanodet.py   — NanoDet-Plus (CLI + pipeline ready, backend pending)
    eval_picodet.py  / infer_picodet.py   — PicoDet (CLI + pipeline ready, backend pending)
```

## Design notes for the client report

- **Box convention**: all boxes are `[x, y, w, h]` top-left based, in original-image
  pixel coordinates (matches Nikon JSON exactly) — model-space predictions are
  transformed back through the inverse letterbox transform before scoring, so
  metrics are computed in the same space as your ground truth regardless of the
  model's input resolution.
- **Recall@FPPI=0.05** is computed by sweeping the confidence threshold and
  finding the recall value at the point where FP/image first reaches 0.05 —
  standard for this metric, exposed as a full curve (`recall_vs_fppi.png`) in
  every report for transparency, not just the single operating point.
- **Sports-wise recall** uses the same operating point selection independently
  per sport subset, since the client wants this for per-sport development
  planning rather than as a single blended number.
