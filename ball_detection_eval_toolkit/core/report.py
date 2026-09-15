"""
Report assembly matching client's "Numbers/others Expected in Report" slide:
    1. Recall @ FPPI=0.05 (main criteria)
    2. Sports-wise recall
    3. GMACs from torchinfo summary
    4. Loss curves (selected models only)
    5. FP/FN samples (selected models only)
"""

import json
from pathlib import Path
from typing import Optional

import numpy as np


def compute_gmacs_torchinfo(model, input_size=(1, 3, 320, 480)) -> dict:
    """
    Requires a PyTorch nn.Module (works for NanoDet-Plus .pt models directly;
    NOT applicable to ONNX or Paddle -- for those, report GMACs from the
    training-framework equivalent model if available, or convert to a torch
    module solely for this measurement).
    """
    try:
        from torchinfo import summary
    except ImportError:
        raise ImportError("pip install torchinfo --break-system-packages")

    import torch
    model.eval()
    with torch.no_grad():
        stats = summary(model, input_size=input_size, verbose=0)
    # torchinfo reports MACs as total_mult_adds
    macs = stats.total_mult_adds
    gmacs = macs / 1e9
    return {"gmacs": gmacs, "total_params": stats.total_params, "raw_summary": str(stats)}


def plot_loss_curve(log_values: dict, out_path: str, title: str = "Training Loss"):
    """
    log_values: {"train_loss": [...], "val_loss": [...] (optional)}
    Writes a PNG to out_path. Requires matplotlib.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    for key, values in log_values.items():
        ax.plot(values, label=key)
    ax.set_xlabel("Epoch / Step")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_fppi_recall_curve(curve: dict, out_path: str, target_fppi: float,
                            title: str = "Recall vs FPPI"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(curve["fppi"], curve["recall"])
    ax.axvline(target_fppi, color="red", linestyle="--", label=f"target FPPI={target_fppi}")
    ax.set_xlabel("False Positives Per Image (FPPI)")
    ax.set_ylabel("Recall")
    ax.set_title(title)
    ax.set_xlim(0, max(0.2, target_fppi * 4))
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def draw_fp_fn_overlays(fp_fn_data: dict, image_root: str, out_dir: str, image_id_to_path):
    """
    image_id_to_path: callable(image_id) -> absolute image path
    Draws GT (green) and FP (red) boxes on copies of the relevant images.
    """
    import cv2

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    by_image = {}
    for fp in fp_fn_data.get("false_positives", []):
        by_image.setdefault(fp["image_id"], {"fp": [], "fn": []})["fp"].append(fp["box_xyxy"])
    for fn in fp_fn_data.get("false_negatives", []):
        by_image.setdefault(fn["image_id"], {"fp": [], "fn": []})["fn"].append(fn["gt_box_xyxy"])

    saved = []
    for image_id, boxes in by_image.items():
        img_path = image_id_to_path(image_id)
        img = cv2.imread(img_path)
        if img is None:
            continue
        for b in boxes["fp"]:
            x1, y1, x2, y2 = [int(v) for v in b]
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(img, "FP", (x1, max(0, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        for b in boxes["fn"]:
            x1, y1, x2, y2 = [int(v) for v in b]
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, "FN (missed GT)", (x1, max(0, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        safe_name = image_id.replace("/", "__") + "_fpfn.jpg"
        out_path = out_dir / safe_name
        cv2.imwrite(str(out_path), img)
        saved.append(str(out_path))

    return saved


def write_markdown_report(eval_result: dict, model_name: str, out_path: str,
                           gmacs_info: Optional[dict] = None,
                           recall_curve_png: Optional[str] = None,
                           loss_curve_png: Optional[str] = None,
                           fp_fn_dir: Optional[str] = None):
    lines = [f"# Evaluation Report — {model_name}", ""]
    lines.append(f"- IoU threshold: {eval_result['iou_threshold']}")
    lines.append(f"- Target FPPI: {eval_result['target_fppi']}")
    lines.append(f"- Total images: {eval_result['n_images_total']}")
    lines.append(f"- Total GT boxes: {eval_result['n_gt_total']}")
    lines.append("")

    lines.append("## 1. Recall @ FPPI=0.05 (Main Evaluation Criteria)")
    ov = eval_result["overall"]
    lines.append(f"- **Recall: {ov['recall_at_target_fppi']:.4f}**")
    lines.append(f"- Operating confidence threshold: {ov['operating_threshold']}")
    lines.append(f"- Meets >60% target: {'YES' if ov['recall_at_target_fppi'] > 0.6 else 'NO'}")
    lines.append("")

    lines.append("## 2. Sports-wise Recall")
    lines.append("| Sport | Recall@FPPI=0.05 | # GT boxes | # Images |")
    lines.append("|---|---|---|---|")
    for sport, res in eval_result["per_sport"].items():
        lines.append(f"| {sport} | {res['recall_at_target_fppi']:.4f} | {res['n_gt']} | {res['n_images']} |")
    lines.append("")

    if gmacs_info:
        lines.append("## 3. GMACs (torchinfo)")
        lines.append(f"- GMACs: {gmacs_info['gmacs']:.4f}")
        lines.append(f"- Total params: {gmacs_info['total_params']:,}")
        lines.append(f"- Meets <1 GOPS target: {'YES' if gmacs_info['gmacs'] < 1.0 else 'CHECK (spec says GOPs, confirm units with client)'}")
        lines.append("")

    if recall_curve_png:
        lines.append("## Recall vs FPPI curve")
        lines.append(f"![recall curve]({recall_curve_png})")
        lines.append("")

    if loss_curve_png:
        lines.append("## 4. Loss Curve")
        lines.append(f"![loss curve]({loss_curve_png})")
        lines.append("")

    if fp_fn_dir:
        lines.append("## 5. False Positive / False Negative Samples")
        lines.append(f"See images in: `{fp_fn_dir}`")
        lines.append("")

    Path(out_path).write_text("\n".join(lines), encoding="utf-8")

    # also dump raw JSON for programmatic consumption
    json_path = str(Path(out_path).with_suffix(".json"))
    with open(json_path, "w") as f:
        json.dump(eval_result, f, indent=2)

    return out_path, json_path
