python - <<'PY'
import torch

ckpt = torch.load("your_model.pth", map_location="cpu", weights_only=False)

print("Type:", type(ckpt))

if isinstance(ckpt, dict):
    print("\nKeys:")
    for k, v in ckpt.items():
        print(f"  {k}: {type(v)}")
else:
    print(ckpt)
PY

python calc_gmac_thop.py \
  --config /home/test_mohit/workspace_mohit/nanodet/config/sep7/YOUR_CORRECT_CONFIG.yml \
  --model /home/test_mohit/workspace_mohit/nanodet/trained_models/sep7/ball_sep7_shf_ch_48/model_best/nanodet_model_best.pth \
  --height 320 \
  --width 480
