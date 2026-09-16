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
