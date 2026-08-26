import json
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image


DATASET = Path("/home/eng_megha/balldataset_coco")

SPLIT = "train"
NUM_IMAGES = 10

OUTPUT = DATASET / "visualizations"
OUTPUT.mkdir(exist_ok=True)


# Load COCO
with open(
    DATASET / "annotations" / f"instances_{SPLIT}.json",
    "r",
    encoding="utf-8"
) as f:
    coco = json.load(f)


# Image ID -> annotations
anns = {}

for ann in coco["annotations"]:
    anns.setdefault(ann["image_id"], []).append(ann)


# Random images
images = random.sample(
    coco["images"],
    min(NUM_IMAGES, len(coco["images"]))
)


for i, info in enumerate(images):

    image_path = (
        DATASET /
        "images" /
        SPLIT /
        info["file_name"]
    )

    if not image_path.exists():
        print("Missing:", image_path)
        continue

    image = Image.open(image_path).convert("RGB")

    fig, ax = plt.subplots(figsize=(10, 8))

    ax.imshow(image)

    for ann in anns.get(info["id"], []):

        x, y, w, h = ann["bbox"]

        rect = patches.Rectangle(
            (x, y),
            w,
            h,
            linewidth=2,
            edgecolor="red",
            facecolor="none",
        )

        ax.add_patch(rect)

    ax.set_title(
        f"Image {i + 1} | Boxes: "
        f"{len(anns.get(info['id'], []))}"
    )

    ax.axis("off")

    output_file = (
        OUTPUT /
        f"{SPLIT}_{i + 1}.jpg"
    )

    plt.savefig(
        output_file,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()

    print("Saved:", output_file)


print("\nDone.")
print("Open:", OUTPUT)
