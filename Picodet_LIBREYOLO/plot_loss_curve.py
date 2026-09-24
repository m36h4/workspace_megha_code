"""Plot the loss curve (and validation mAP, if logged) from a LibreYOLO training run.

LibreYOLO writes runs/train/<name>/metrics.jsonl automatically -- one JSON line per
epoch, no need to keep train()'s return value around. This just reads that file.

Usage:
    python plot_loss_curve.py runs/train/lcnet_exp/metrics.jsonl
    python plot_loss_curve.py runs/train/lcnet_exp/metrics.jsonl --out loss.png
    python plot_loss_curve.py run1/metrics.jsonl run2/metrics.jsonl --labels ESNet LCNet
"""
import argparse
import json

import matplotlib
matplotlib.use("Agg")  # headless -- writes a file, doesn't try to open a window
import matplotlib.pyplot as plt


def load(path):
    rows = [json.loads(line) for line in open(path) if line.strip()]
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("metrics_jsonl", nargs="+", help="one or more runs/train/<name>/metrics.jsonl files")
    p.add_argument("--labels", nargs="+", default=None, help="legend label per file, same order")
    p.add_argument("--out", default="loss_curve.png")
    p.add_argument("--metric-key", default="metrics/mAP50-95",
                   help="which validation metric to overlay (must match a current_metric_name in the log)")
    args = p.parse_args()

    labels = args.labels or [f.split("/")[-2] if "/" in f else f for f in args.metrics_jsonl]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    for path, label in zip(args.metrics_jsonl, labels):
        rows = load(path)
        epochs = [r["epoch"] for r in rows]

        ax1.plot(epochs, [r["train/loss"] for r in rows], label=f"{label} total")
        for key, style in [("train/cls_loss", "--"), ("train/bbox_loss", ":"), ("train/dfl_loss", "-.")]:
            if key in rows[0]:
                ax1.plot(epochs, [r.get(key, float("nan")) for r in rows], style, alpha=0.6,
                        label=f"{label} {key.split('/')[-1]}")

        val_epochs = [r["epoch"] for r in rows if r.get("validated") and r.get("current_metric") is not None]
        val_vals = [r["current_metric"] for r in rows if r.get("validated") and r.get("current_metric") is not None]
        if val_vals:
            ax2.plot(val_epochs, val_vals, "o-", label=f"{label} ({rows[-1].get('current_metric_name', args.metric_key)})")

    ax1.set_xlabel("epoch"); ax1.set_ylabel("loss"); ax1.set_title("Training loss")
    ax1.legend(fontsize=8); ax1.grid(alpha=0.3)
    ax2.set_xlabel("epoch"); ax2.set_ylabel("metric"); ax2.set_title("Validation metric over training")
    ax2.legend(fontsize=8); ax2.grid(alpha=0.3)
    if not any(ax2.lines):
        ax2.text(0.5, 0.5, "no validation metrics logged\n(train() ran with val_period off\nor no eval happened yet)",
                 ha="center", va="center", transform=ax2.transAxes)

    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
