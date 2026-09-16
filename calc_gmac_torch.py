import torch
from torchinfo import summary
from fvcore.nn import FlopCountAnalysis

model = torch.load("model.pt", map_location="cpu")
model.eval()

x = torch.randn(1, 3, 640, 640)

# torchinfo
summary(model, input_data=x)

# FLOPs
flops = FlopCountAnalysis(model, x).total()

# Usually 1 MAC = 2 FLOPs
gmacs = flops / 2 / 1e9

print(f"GMACs: {gmacs:.3f}")
