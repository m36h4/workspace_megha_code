"""
Rebuild the real LibrePICODET architecture, load your trained checkpoint's
weights into it, and report which activation module classes are present.

Usage:
  python3 check_pt_activations.py runs/train/picodet_exp/weights/last.pt --size s --nb-classes 2
"""
import argparse
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pt_path")
    ap.add_argument("--size", default="s")
    ap.add_argument("--nb-classes", type=int, default=2)
    args = ap.parse_args()

    ckpt = torch.load(args.pt_path, map_location="cpu", weights_only=False)
    print("Checkpoint top-level keys:", list(ckpt.keys()))
    print("model_family:", ckpt.get("model_family"))
    print("libreyolo_version:", ckpt.get("libreyolo_version"))
    print("size:", ckpt.get("size"))
    print()

    state_dict = ckpt["model"]
    if hasattr(state_dict, "state_dict"):
        state_dict = state_dict.state_dict()

    print(f"state_dict has {len(state_dict)} tensors")
    print()

    from libreyolo.models.picodet.nn import LibrePICODETModel
    import libreyolo.models.picodet.nn as nn_mod
    print("libreyolo.models.picodet.nn loaded from:", nn_mod.__file__)
    print()

    model = LibrePICODETModel(size=args.size, nb_classes=args.nb_classes)

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"load_state_dict: {len(missing)} missing keys, {len(unexpected)} unexpected keys")
    if missing:
        print("  sample missing:", missing[:5])
    if unexpected:
        print("  sample unexpected:", unexpected[:5])
    print()

    act_counts = {}
    for m in model.modules():
        cls_name = m.__class__.__name__
        if cls_name in ("Hardswish", "SiLU", "Sigmoid", "HSigmoid", "ReLU", "LeakyReLU", "Identity"):
            act_counts[cls_name] = act_counts.get(cls_name, 0) + 1

    print("Activation module counts in the RECONSTRUCTED model (from currently installed libreyolo code):")
    for k, v in sorted(act_counts.items()):
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
