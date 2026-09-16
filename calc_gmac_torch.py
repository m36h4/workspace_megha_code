import torch
from nanodet.util import cfg, load_config, Logger, load_model_weight
from nanodet.model.arch import build_model
from fvcore.nn import FlopCountAnalysis


CONFIG = "/home/test_mohit/workspace_mohit/nanodet/config/sep3/ball_sep3_nanodet_320x480_shf_1.yml"

CHECKPOINT = "/home/test_mohit/workspace_mohit/nanodet/trained_models/sep7/ball_sep7_shf_ch_48/model_best/nanodet_model_best.pth"


# --------------------------------------------------
# Load config
# --------------------------------------------------
load_config(cfg, CONFIG)

logger = Logger(-1, cfg.save_dir, False)

# --------------------------------------------------
# Build model from config
# --------------------------------------------------
model = build_model(cfg.model)

# --------------------------------------------------
# Load checkpoint
# --------------------------------------------------
checkpoint = torch.load(
    CHECKPOINT,
    map_location="cpu",
    weights_only=False
)

load_model_weight(model, checkpoint, logger)

model.eval()
model.cpu()

# --------------------------------------------------
# Input: 320 x 480
# --------------------------------------------------
dummy = torch.randn(1, 3, 320, 480)

# --------------------------------------------------
# Parameters
# --------------------------------------------------
params = sum(p.numel() for p in model.parameters())

# --------------------------------------------------
# FLOPs / MACs
# --------------------------------------------------
with torch.no_grad():
    flops = FlopCountAnalysis(model, dummy).total()

gflops = flops / 1e9
gmacs = flops / 2 / 1e9

print()
print("=" * 50)
print("NanoDet GMAC Calculation")
print("=" * 50)
print(f"Input       : 1 x 3 x 320 x 480")
print(f"Parameters  : {params:,}")
print(f"Params (M)  : {params / 1e6:.3f}")
print(f"GFLOPs      : {gflops:.3f}")
print(f"GMACs       : {gmacs:.3f}")
print("=" * 50)
