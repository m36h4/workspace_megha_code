"""
Full training script for PicoDet (LibreYOLO), with:
  - Rectangular letterbox preprocessing at 320x480 (train + val)
  - Swish/Sigmoid activation architecture (already patched into
    libreyolo/models/picodet/nn.py on this machine)
  - From-scratch initialization (no pretrained weights, since the
    activation swap makes pretrained checkpoints numerically incompatible)

Requires letterbox_picodet_rect.py in the same directory (or on PYTHONPATH).

Usage:
  python3 train_picodet.py                # full run
  python3 train_picodet.py --smoke        # 2-epoch sanity check first
  python3 train_picodet.py --pretrained   # NOT recommended, see warning below
"""
import argparse
import sys

# --- Apply the letterbox patch BEFORE importing/building the model ---
# These monkey-patches must happen before LibrePICODET is constructed,
# since the trainer looks up PICODETTrainTransform / PICODETValPreprocessor
# by name from these modules at train-setup time.
import libreyolo.models.picodet.trainer as picodet_trainer
import libreyolo.validation.preprocessors as val_preproc
from letterbox_picodet_rect import (
    LetterboxPICODETTrainTransform,
    LetterboxPICODETValPreprocessor,
)

picodet_trainer.PICODETTrainTransform = LetterboxPICODETTrainTransform
val_preproc.PICODETValPreprocessor = LetterboxPICODETValPreprocessor

from libreyolo.models.picodet.model import LibrePICODET


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset.yaml", help="Path to dataset yaml")
    ap.add_argument("--size", default="s", choices=["s", "m", "l"], help="PicoDet size")
    ap.add_argument("--nb-classes", type=int, default=2, help="Number of classes")
    ap.add_argument("--epochs", type=int, default=300, help="Training epochs")
    ap.add_argument("--batch", type=int, default=64, help="Batch size")
    ap.add_argument("--imgsz-h", type=int, default=320, help="Input height")
    ap.add_argument("--imgsz-w", type=int, default=480, help="Input width")
    ap.add_argument("--lr0", type=float, default=0.1,
                     help="Initial LR. 0.1 for from-scratch (matches upstream PicoDet's "
                          "randomly-initialized-weights regime); use 0.01 only when "
                          "fine-tuning from a compatible pretrained checkpoint.")
    ap.add_argument("--pretrained", action="store_true",
                     help="Load pretrained weights (NOT recommended here -- pretrained "
                          "checkpoints were trained with HardSwish/HardSigmoid, "
                          "incompatible with the Swish/Sigmoid architecture patch)")
    ap.add_argument("--project", default="runs/train", help="Output project dir")
    ap.add_argument("--name", default="picodet_exp", help="Run name")
    ap.add_argument("--device", default="auto", help="'cpu', 'cuda', or 'auto'")
    ap.add_argument("--workers", type=int, default=4, help="Dataloader workers")
    ap.add_argument("--patience", type=int, default=50, help="Early stopping patience")
    ap.add_argument("--loggers", default=None, help="e.g. 'tensorboard' or 'wandb'")
    ap.add_argument("--smoke", action="store_true",
                     help="Override epochs/batch for a quick 2-epoch sanity run")
    args = ap.parse_args()

    if args.smoke:
        print(">>> SMOKE TEST MODE: epochs=2, batch=16, workers=0 <<<")
        args.epochs = 2
        args.batch = 16
        args.workers = 0

    if args.pretrained:
        print(
            "WARNING: --pretrained was passed, but this architecture uses Swish/Sigmoid "
            "instead of the original HardSwish/HardSigmoid. Loading pretrained weights "
            "will not error, but the checkpoint's weights were optimized for a different "
            "nonlinearity and will likely hurt more than help. Proceeding anyway since "
            "you explicitly requested it.",
            file=sys.stderr,
        )

    print(f"Building LibrePICODET(size={args.size!r}, nb_classes={args.nb_classes})")
    model = LibrePICODET(size=args.size, nb_classes=args.nb_classes)

    print(
        f"Starting training: imgsz=({args.imgsz_h},{args.imgsz_w}) epochs={args.epochs} "
        f"batch={args.batch} lr0={args.lr0} pretrained={args.pretrained}"
    )
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=(args.imgsz_h, args.imgsz_w),
        batch=args.batch,
        lr0=args.lr0,
        pretrained=args.pretrained,
        device=args.device,
        workers=args.workers,
        project=args.project,
        name=args.name,
        patience=args.patience,
        loggers=args.loggers,
    )

    print("\nTraining finished.")
    print(results)


if __name__ == "__main__":
    main()
