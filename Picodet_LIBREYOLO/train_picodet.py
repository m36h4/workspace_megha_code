"""Fine-tune LibrePICODETs on a YOLO-format dataset (CPU or GPU).

Usage:
    python train_picodet.py --data my-dataset.yaml                 # auto (GPU if available)
    python train_picodet.py --data my-dataset.yaml --device cpu
    python train_picodet.py --data my-dataset.yaml --device 0      # first CUDA GPU
    python train_picodet.py --data coco8.yaml --epochs 3 --batch 4 # quick smoke test
"""
import argparse

import torch
from libreyolo import LibreYOLO


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="dataset YAML (path or bundled name like coco8.yaml)")
    p.add_argument("--weights", default="LibrePICODETs.pt")  # s=320px, m=416px, l=640px
    p.add_argument("--device", default="", help="'' = auto, 'cpu', '0', '0,1' ...")
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--lr0", type=float, default=0.01)
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()

    print("torch", torch.__version__, "| CUDA available:", torch.cuda.is_available())

    model = LibreYOLO(args.weights)  # downloads from Hugging Face on first use

    results = model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        lr0=args.lr0,
        device=args.device,
        workers=args.workers,
        # imgsz left unset: Python API uses the checkpoint's native size (320 for s)
        # amp defaults to True; it only applies on CUDA, so it's harmless on CPU
    )
    print(results)

    metrics = model.val(data=args.data)
    print("mAP50-95:", metrics["metrics/mAP50-95"], "| mAP50:", metrics["metrics/mAP50"])


if __name__ == "__main__":  # guard needed for dataloader workers / multi-GPU spawn
    main()
