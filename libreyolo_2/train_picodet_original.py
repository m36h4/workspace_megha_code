import argparse
import os

from libreyolo.models.picodet.model import LibrePICODET


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--data", default="dataset/data.yaml")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=32)

    # Your actual input size
    ap.add_argument("--imgsz", type=int, default=320)

    # Converted upstream PicoDet-S weights
    ap.add_argument(
        "--weights",
        default="weights/picodet_s_320.pt",
        help="Converted PicoDet-S pretrained weights",
    )

    ap.add_argument("--device", default="auto")
    ap.add_argument("--workers", type=int, default=4)

    ap.add_argument("--project", default="runs/train")
    ap.add_argument("--name", default="picodet_pretrained_320")

    ap.add_argument("--patience", type=int, default=30)

    args = ap.parse_args()

    print("=" * 60)
    print("Original PicoDet-S pretrained fine-tuning")
    print("=" * 60)

    print(f"Dataset     : {args.data}")
    print(f"Weights     : {args.weights}")
    print(f"Image size  : {args.imgsz}x{args.imgsz}")
    print(f"Epochs      : {args.epochs}")
    print(f"Batch       : {args.batch}")
    print(f"Device      : {args.device}")
    print("=" * 60)

    if not os.path.isfile(args.weights):
        raise FileNotFoundError(
            f"\nPretrained weights not found:\n{args.weights}\n\n"
            "Convert the upstream PicoDet-S checkpoint first."
        )

    # Original PicoDet architecture.
    # DO NOT use the Swish/Sigmoid-patched version.
    model = LibrePICODET(
        size="s",
        nb_classes=2,
    )

    print("\nStarting fine-tuning...")

    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,

        # Fine-tuning from converted pretrained weights
        pretrained=args.weights,

        device=args.device,
        workers=args.workers,

        project=args.project,
        name=args.name,

        patience=args.patience,
    )

    print("\nTraining finished.")
    print(results)


if __name__ == "__main__":
    main()
