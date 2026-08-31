conda create -n libreyolo_gpu python=3.12 -y
conda activate libreyolo_gpu

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130

python3 -c "
import torch
print('torch version:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
print('CUDA version (torch built with):', torch.version.cuda)
print('Device name:', torch.cuda.get_device_name(0))
print('Compute capability:', torch.cuda.get_device_capability(0))

# The real test -- an actual op on the GPU, not just is_available()
x = torch.randn(1000, 1000, device='cuda')
y = x @ x
torch.cuda.synchronize()
print('Matrix multiply on GPU: OK, result sum =', y.sum().item())
"

cd ~/libreyolo
pip install -e .

python3 -c "
from libreyolo.models.picodet.model import LibrePICODETModel
model = LibrePICODETModel(size='s', nb_classes=2)
act_counts = {}
for m in model.modules():
    n = m.__class__.__name__
    if n in ('SiLU','Sigmoid','Hardswish','HSigmoid','ReLU','Identity'):
        act_counts[n] = act_counts.get(n,0)+1
print(act_counts)
"

python3 train_picodet.py --smoke --device cuda --batch 32
