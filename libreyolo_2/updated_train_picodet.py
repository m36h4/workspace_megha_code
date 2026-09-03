"""
PicoDet-S training with:

- Pretrained LibrePICODETs.pt initialization
- Swish / Sigmoid activation patch
- Rectangular 320x480 input
- Custom PicoDet letterbox preprocessing
- Ball detection dataset
- Loss curve output

IMPORTANT:
The Swish/Sigmoid patch must already be applied in:
    libreyolo/models/picodet/nn.py

The rectangular preprocessing must be available as:
    letterbox_picodet_rect.py
"""

import argparse
import os
import sys

# ============================================================
# 1. Apply custom preprocessing BEFORE constructing the model
# ============================================================

import libreyolo.models.picodet.trainer as picodet_trainer
import libreyolo.validation.preprocessors as val_preproc

from letterbox_picodet_rect import (
    LetterboxPICODETTrainTransform,
    LetterboxPICODETValPreprocessor,
)

picodet_trainer.PICODETTrainTransform = LetterboxPICODETTrainTransform
val_preproc.PICODETValPreprocessor = LetterboxPICODETValPreprocessor


# ============================================================
# 2. LibreYOLO model factory
# ============================================================

from libreyolo import LibreYOLO


def main():

    ap = argparse.ArgumentParser()

    # -----------------------------
    # Dataset
    # -----------------------------
    ap.add_argument(
        "--data",
        default="dataset.yaml",
        help="Path to YOLO dataset.yaml"
    )

    # -----------------------------
    # Model
    # -----------------------------
    ap.add_argument(
        "--weights",
        default="LibrePICODETs.pt",
        help="Pretrained PicoDet-S checkpoint"
    )

    # -----------------------------
    # Training
    # -----------------------------
    ap.add_argument(
        "--epochs",
        type=int,
        default=100
    )

    ap.add_argument(
        "--batch",
        type=int,
        default=64
    )

    ap.add_argument(
        "--lr0",
        type=float,
        default=0.001
    )

    ap.add_argument(
        "--workers",
        type=int,
        default=8
    )

    ap.add_argument(
        "--patience",
        type=int,
        default=30
    )

    # -----------------------------
    # Input size
    # -----------------------------
    ap.add_argument(
        "--imgsz-h",
        type=int,
        default=320
    )

    ap.add_argument(
        "--imgsz-w",
        type=int,
        default=480
    )

    # -----------------------------
    # Runtime
    # -----------------------------
    ap.add_argument(
        "--device",
        default="auto",
        help="'cpu', 'cuda', '0', or 'auto'"
    )

    ap.add_argument(
        "--project",
        default="runs/train"
    )

    ap.add_argument(
        "--name",
        default="picodet_pretrained_swish_320x480"
    )

    ap.add_argument(
        "--loggers",
        default=None,
        help="tensorboard / wandb / etc."
    )

    ap.add_argument(
        "--smoke",
        action="store_true",
        help="2 epoch sanity test"
    )

    args = ap.parse_args()


    # ========================================================
    # Smoke test
    # ========================================================

    if args.smoke:

        print(
            ">>> SMOKE TEST: epochs=2, batch=16, workers=0 <<<"
        )

        args.epochs = 2
        args.batch = 16
        args.workers = 0


    # ========================================================
    # Print configuration
    # ========================================================

    print("\n==============================================")
    print("PicoDet Training")
    print("==============================================")

    print(f"Weights       : {args.weights}")
    print(f"Dataset       : {args.data}")
    print(f"Input         : {args.imgsz_h} x {args.imgsz_w}")
    print(f"Epochs        : {args.epochs}")
    print(f"Batch         : {args.batch}")
    print(f"Learning rate : {args.lr0}")
    print(f"Workers       : {args.workers}")
    print(f"Patience      : {args.patience}")
    print(f"Device        : {args.device}")

    print("----------------------------------------------")
    print("Activation    : Swish / Sigmoid")
    print("Initialization: PRETRAINED PicoDet-S")
    print("==============================================\n")


    # ========================================================
    # Check checkpoint
    # ========================================================

    if not os.path.exists(args.weights):

        print(
            f"Checkpoint '{args.weights}' was not found locally.",
            file=sys.stderr
        )

        print(
            "LibreYOLO can download published weights automatically "
            "when using the model name."
        )


    # ========================================================
    # Construct PRETRAINED PicoDet
    #
    # IMPORTANT:
    # This replaces:
    #
    #   LibrePICODET(...)
    #
    # and:
    #
    #   pretrained=True
    #
    # ========================================================

    print("Loading pretrained PicoDet-S...")

    model = LibreYOLO(args.weights)

    print("Pretrained PicoDet-S loaded.")


    # ========================================================
    # Training
    # ========================================================

    print("\nStarting training...\n")

    results = model.train(

        data=args.data,

        epochs=args.epochs,

        batch=args.batch,

        imgsz=(args.imgsz_h, args.imgsz_w),

        lr0=args.lr0,

        device=args.device,

        workers=args.workers,

        project=args.project,

        name=args.name,

        patience=args.patience,

        loggers=args.loggers,

    )


    # ========================================================
    # Training finished
    # ========================================================

    print("\n==============================================")
    print("Training finished")
    print("==============================================")

    print(results)


    # ========================================================
    # Loss curve
    # ========================================================

    try:

        import matplotlib.pyplot as plt

        losses = results.get("epoch_losses", [])

        if losses:

            plt.figure(figsize=(8, 5))

            plt.plot(
                range(1, len(losses) + 1),
                losses
            )

            plt.xlabel("Epoch")
            plt.ylabel("Training Loss")

            plt.title(
                "PicoDet-S Training Loss "
                "(Swish/Sigmoid, 320×480)"
            )

            plt.grid(True)

            plt.tight_layout()

            loss_dir = os.path.join(
                args.project,
                args.name
            )

            os.makedirs(
                loss_dir,
                exist_ok=True
            )

            loss_path = os.path.join(
                loss_dir,
                "loss_curve.png"
            )

            plt.savefig(
                loss_path,
                dpi=200
            )

            plt.close()

            print(
                f"\nLoss curve saved to:\n{loss_path}"
            )

        else:

            print(
                "\nNo epoch_losses found in training results."
            )

    except Exception as e:

        print(
            f"\nCould not create loss curve: {e}"
        )


if __name__ == "__main__":
    main()
