import argparse
import torch

from nanodet.util import cfg, load_config, Logger, load_model_weight
from nanodet.model.arch import build_model
from fvcore.nn import FlopCountAnalysis


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        required=True,
        help="Path to NanoDet config YAML"
    )

    parser.add_argument(
        "--model",
        required=True,
        help="Path to .pth checkpoint"
    )

    parser.add_argument(
        "--height",
        type=int,
        default=320
    )

    parser.add_argument(
        "--width",
        type=int,
        default=480
    )

    return parser.parse_args()


def count_parameters(model):
    total = 0
    aux = 0
    main = 0

    for name, param in model.named_parameters():

        n = param.numel()
        total += n

        if "aux" in name.lower():
            aux += n
        else:
            main += n

    return total, main, aux


def main():

    args = parse_args()

    # ---------------------------------------------------------
    # CONFIG
    # ---------------------------------------------------------

    print("Loading config...")
    print(f"Config: {args.config}")

    load_config(cfg, args.config)

    # ---------------------------------------------------------
    # BUILD MODEL
    # ---------------------------------------------------------

    print("\nBuilding NanoDet model...")

    model = build_model(cfg.model)

    # ---------------------------------------------------------
    # LOAD CHECKPOINT
    # ---------------------------------------------------------

    print("Loading checkpoint...")
    print(f"Model: {args.model}")

    checkpoint = torch.load(
        args.model,
        map_location="cpu",
        weights_only=False
    )

    logger = Logger(-1, cfg.save_dir, False)

    load_model_weight(
        model,
        checkpoint,
        logger
    )

    # ---------------------------------------------------------
    # EVAL MODE
    # ---------------------------------------------------------

    model.eval()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = model.to(device)

    print(f"\nDevice: {device}")

    # ---------------------------------------------------------
    # INPUT
    # ---------------------------------------------------------

    dummy_input = torch.randn(
        1,
        3,
        args.height,
        args.width,
        device=device
    )

    print(
        f"Input: 1 x 3 x "
        f"{args.height} x {args.width}"
    )

    # ---------------------------------------------------------
    # PARAMETERS
    # ---------------------------------------------------------

    total_params, main_params, aux_params = (
        count_parameters(model)
    )

    # ---------------------------------------------------------
    # FVCORE FLOPs
    # ---------------------------------------------------------

    print("\nCalculating FLOPs...")

    with torch.no_grad():

        flops_analysis = FlopCountAnalysis(
            model,
            dummy_input
        )

        total_flops = flops_analysis.total()

    # ---------------------------------------------------------
    # UNSUPPORTED OPERATORS
    # ---------------------------------------------------------

    unsupported = flops_analysis.unsupported_ops()

    # ---------------------------------------------------------
    # MACS / FLOPs
    # ---------------------------------------------------------

    total_macs = total_flops / 2

    gmacs = total_macs / 1e9
    gflops = total_flops / 1e9

    # ---------------------------------------------------------
    # OUTPUT
    # ---------------------------------------------------------

    print()
    print("=" * 65)
    print("NanoDet Computational Complexity")
    print("=" * 65)

    print(
        f"Input Size              : "
        f"{args.height} x {args.width}"
    )

    print()

    print(
        f"Total Parameters        : "
        f"{total_params:,}"
    )

    print(
        f"Total Parameters (M)    : "
        f"{total_params / 1e6:.6f}"
    )

    print(
        f"Main Parameters         : "
        f"{main_params:,}"
    )

    print(
        f"Main Parameters (M)     : "
        f"{main_params / 1e6:.6f}"
    )

    print(
        f"Aux Parameters          : "
        f"{aux_params:,}"
    )

    print(
        f"Aux Parameters (M)      : "
        f"{aux_params / 1e6:.6f}"
    )

    print()

    print(
        f"Total MACs              : "
        f"{total_macs:,.0f}"
    )

    print(
        f"GMACs                   : "
        f"{gmacs:.6f}"
    )

    print(
        f"Total FLOPs             : "
        f"{total_flops:,.0f}"
    )

    print(
        f"GFLOPs                  : "
        f"{gflops:.6f}"
    )

    print()

    print(
        f"Unsupported Operators   : "
        f"{len(unsupported)}"
    )

    if unsupported:

        print("\nUnsupported operator details:")

        for op, count in unsupported.items():
            print(
                f"  {op}: {count}"
            )

    print("=" * 65)


if __name__ == "__main__":
    main()
