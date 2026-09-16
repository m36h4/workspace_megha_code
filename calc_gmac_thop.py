import argparse
import torch
from thop import profile

from nanodet.model.arch import build_model
from nanodet.util import cfg, load_config, Logger, load_model_weight


def parse_args():
    parser = argparse.ArgumentParser(
        description="Calculate NanoDet parameters, MACs and FLOPs using THOP"
    )

    parser.add_argument(
        "--config",
        required=True,
        help="Path to NanoDet config YAML"
    )

    parser.add_argument(
        "--model",
        required=True,
        help="Path to NanoDet .pth checkpoint"
    )

    parser.add_argument(
        "--height",
        type=int,
        default=320,
        help="Input image height"
    )

    parser.add_argument(
        "--width",
        type=int,
        default=480,
        help="Input image width"
    )

    return parser.parse_args()


# ============================================================
# PARAMETER COUNT
# ============================================================

def count_parameters(model):

    total_params = 0
    main_params = 0
    aux_params = 0

    for name, param in model.named_parameters():

        n = param.numel()

        total_params += n

        if "aux" in name.lower():
            aux_params += n
        else:
            main_params += n

    return total_params, main_params, aux_params


# ============================================================
# CHECKPOINT LOADING
# ============================================================

def load_checkpoint(model, checkpoint_path, logger):

    print("\nLoading checkpoint...")
    print(f"Checkpoint: {checkpoint_path}")

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False
    )

    # NanoDet checkpoint normally contains:
    #
    # {
    #     "state_dict": {...}
    # }
    #
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:

        state_dict = checkpoint["state_dict"]

    elif isinstance(checkpoint, dict):

        state_dict = checkpoint

    else:

        raise TypeError(
            f"Unsupported checkpoint type: {type(checkpoint)}"
        )

    # --------------------------------------------------------
    # Remove possible "model." prefix
    # --------------------------------------------------------

    cleaned_state_dict = {}

    for key, value in state_dict.items():

        if key.startswith("model."):
            key = key[6:]

        cleaned_state_dict[key] = value

    # --------------------------------------------------------
    # Load weights
    # --------------------------------------------------------

    missing, unexpected = model.load_state_dict(
        cleaned_state_dict,
        strict=False
    )

    print(
        f"Missing keys    : {len(missing)}"
    )

    print(
        f"Unexpected keys : {len(unexpected)}"
    )

    if missing:

        print("\nFirst missing keys:")

        for key in missing[:10]:
            print(f"  {key}")

    if unexpected:

        print("\nFirst unexpected keys:")

        for key in unexpected[:10]:
            print(f"  {key}")


# ============================================================
# MAIN
# ============================================================

def main():

    args = parse_args()

    # --------------------------------------------------------
    # CONFIG
    # --------------------------------------------------------

    print("Loading config...")
    print(f"Config: {args.config}")

    load_config(
        cfg,
        args.config
    )

    # --------------------------------------------------------
    # BUILD MODEL
    # --------------------------------------------------------

    print("\nBuilding NanoDet model...")

    model = build_model(
        cfg.model
    )

    # --------------------------------------------------------
    # LOGGER
    # --------------------------------------------------------

    logger = Logger(
        -1,
        cfg.save_dir,
        False
    )

    # --------------------------------------------------------
    # LOAD CHECKPOINT
    # --------------------------------------------------------

    load_checkpoint(
        model,
        args.model,
        logger
    )

    # --------------------------------------------------------
    # EVAL
    # --------------------------------------------------------

    model.eval()

    # --------------------------------------------------------
    # DEVICE
    # --------------------------------------------------------

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = model.to(device)

    print(
        f"\nDevice: {device}"
    )

    # --------------------------------------------------------
    # INPUT
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # PARAMETERS
    # --------------------------------------------------------

    (
        total_params,
        main_params,
        aux_params
    ) = count_parameters(model)

    # --------------------------------------------------------
    # THOP
    # --------------------------------------------------------

    print("\nCalculating MACs with THOP...")

    with torch.no_grad():

        macs, thop_params = profile(
            model,
            inputs=(dummy_input,),
            verbose=False
        )

    # --------------------------------------------------------
    # CONVERT
    # --------------------------------------------------------

    gmacs = macs / 1e9

    # Standard convention:
    #
    # 1 MAC = 2 FLOPs
    #

    flops = macs * 2

    gflops = flops / 1e9

    # THOP's own parameter count
    thop_params_m = thop_params / 1e6

    # Our parameter count
    total_params_m = total_params / 1e6
    main_params_m = main_params / 1e6
    aux_params_m = aux_params / 1e6

    # --------------------------------------------------------
    # OUTPUT
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("NanoDet Computational Complexity - THOP")
    print("=" * 70)

    print(
        f"Config                  : {args.config}"
    )

    print(
        f"Checkpoint              : {args.model}"
    )

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
        f"{total_params_m:.6f}"
    )

    print()

    print(
        f"Main Parameters         : "
        f"{main_params:,}"
    )

    print(
        f"Main Parameters (M)     : "
        f"{main_params_m:.6f}"
    )

    print()

    print(
        f"Aux Parameters          : "
        f"{aux_params:,}"
    )

    print(
        f"Aux Parameters (M)      : "
        f"{aux_params_m:.6f}"
    )

    print()

    print(
        f"THOP Parameters         : "
        f"{thop_params:,.0f}"
    )

    print(
        f"THOP Parameters (M)     : "
        f"{thop_params_m:.6f}"
    )

    print()

    print(
        f"MACs                    : "
        f"{macs:,.0f}"
    )

    print(
        f"MACs (M)                : "
        f"{macs / 1e6:.6f}"
    )

    print(
        f"GMACs                   : "
        f"{gmacs:.6f}"
    )

    print()

    print(
        f"FLOPs                   : "
        f"{flops:,.0f}"
    )

    print(
        f"GFLOPs                  : "
        f"{gflops:.6f}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()
