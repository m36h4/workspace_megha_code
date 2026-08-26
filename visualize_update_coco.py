import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image


# ============================================================
# CHANGE THIS IF NEEDED
# ============================================================

DATASET = Path("/home/eng_megha/balldataset_coco")

SPLIT = "train"       # train / val / test
NUM_IMAGES = 10


# ============================================================
# LOAD COCO
# ============================================================

json_path = (
    DATASET /
    "annotations" /
    f"instances_{SPLIT}.json"
)

image_dir = (
    DATASET /
    "images" /
    SPLIT
)


with open(
    json_path,
    "r",
    encoding="utf-8"
) as f:

    coco = json.load(f)


# ============================================================
# INDEX ANNOTATIONS
# ============================================================

annotations = {}

for ann in coco["annotations"]:

    image_id = ann["image_id"]

    if image_id not in annotations:
        annotations[image_id] = []

    annotations[image_id].append(ann)


# ============================================================
# RANDOM IMAGES
# ============================================================

images = coco["images"]

random.shuffle(images)

images = images[
    :min(NUM_IMAGES, len(images))
]


# ============================================================
# VISUALIZE
# ============================================================

for img_info in images:

    image_id = img_info["id"]

    filename = img_info["file_name"]

    image_path = (
        image_dir /
        filename
    )

    if not image_path.exists():

        print(
            f"Missing image: {image_path}"
        )

        continue

    image = Image.open(
        image_path
    ).convert("RGB")

    fig, ax = plt.subplots(
        figsize=(10, 8)
    )

    ax.imshow(image)

    # --------------------------------------------------------
    # Draw boxes
    # --------------------------------------------------------

    anns = annotations.get(
        image_id,
        []
    )

    for ann in anns:

        x, y, w, h = ann["bbox"]

        box = patches.Rectangle(
            (x, y),
            w,
            h,
            linewidth=2,
            edgecolor="red",
            facecolor="none",
        )

        ax.add_patch(box)

        ax.text(
            x,
            max(0, y - 5),
            "ball",
            fontsize=10,
            color="red",
            backgroundcolor="white",
        )

    ax.set_title(
        f"{SPLIT}: {filename}\n"
        f"Boxes: {len(anns)}"
    )

    ax.axis("off")

    plt.tight_layout()

    plt.show()
