#!/usr/bin/env python3
"""Diagnose a PicoDet checkpoint/library architecture mismatch.

Run this on the SAME machine/env where training happened, pointing at your
installed libreyolo (the one at /home/megha/libreyolo) and the checkpoint.

    python check_arch_mismatch.py --weights .../last.pt
"""
import argparse
import inspect
import subprocess
import torch

p = argparse.ArgumentParser()
p.add_argument("--weights", required=True)
args = p.parse_args()

ckpt = torch.load(args.weights, map_location="cpu", weights_only=False)
sd = ckpt["model"] if "model" in ckpt else ckpt

print("=== Checkpoint metadata ===")
for k in ("size", "nc", "model_family", "libreyolo_version", "schema_version"):
    print(f"  {k}: {ckpt.get(k) if isinstance(ckpt, dict) else 'N/A'}")

print("\n=== Checkpoint stem/neck shapes ===")
for key in ("backbone.conv1.conv.weight", "neck.trans.0.conv.weight"):
    if key in sd:
        print(f"  {key}: {tuple(sd[key].shape)}")
    else:
        print(f"  {key}: MISSING")

print("\n=== Full checkpoint backbone.blocks.0 / conv1 related keys+shapes ===")
for k, v in sd.items():
    if k.startswith("backbone.conv1") or k.startswith("backbone.blocks.0."):
        print(f"  {k}: {tuple(v.shape)}")

print("\n=== Installed libreyolo ESNet.ARCH ===")
from libreyolo.models.picodet.nn import ESNet
nn_file = inspect.getfile(ESNet)
print(f"  libreyolo module file: {nn_file}")
print(f"  ESNet.ARCH keys: {list(ESNet.ARCH.keys())}")
for size, cfg in ESNet.ARCH.items():
    print(f"    {size}: scale={cfg['scale']}")

print("\n=== Literal stem definition (stage_channels[0]) from source ===")
with open(nn_file) as f:
    lines = f.readlines()
for i, line in enumerate(lines):
    if "stage_channels" in line and "=" in line:
        # print a window of context around each match
        start = max(0, i - 2)
        end = min(len(lines), i + 6)
        print(f"  --- around line {i+1} ---")
        for j in range(start, end):
            print(f"  {j+1:4d}: {lines[j].rstrip()}")
        print()

print("=== Git status/log for this file (if it's a git repo) ===")
repo_dir = "/home/megha/libreyolo"
try:
    log = subprocess.run(
        ["git", "-C", repo_dir, "log", "--oneline", "-n", "10", "--", "libreyolo/models/picodet/nn.py"],
        capture_output=True, text=True, timeout=10,
    )
    print("  git log (last 10 commits touching nn.py):")
    print("  " + (log.stdout.strip().replace("\n", "\n  ") or "(no output / not tracked)"))
    if log.stderr.strip():
        print("  stderr:", log.stderr.strip())

    diff = subprocess.run(
        ["git", "-C", repo_dir, "diff", "HEAD", "--", "libreyolo/models/picodet/nn.py"],
        capture_output=True, text=True, timeout=10,
    )
    if diff.stdout.strip():
        print("\n  UNCOMMITTED local changes to nn.py:")
        print("  " + diff.stdout.strip().replace("\n", "\n  "))
    else:
        print("\n  No uncommitted changes to nn.py (working tree clean for this file).")
except Exception as e:
    print(f"  git check failed or not a git repo: {e}")

import libreyolo
print(f"\nlibreyolo.__version__ (installed): {getattr(libreyolo, '__version__', 'unknown')}")
print(f"libreyolo package location: {inspect.getfile(libreyolo)}")
