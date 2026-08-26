import json
import random
from pathlib import Path

from PIL import Image, ImageDraw


# ============================================================
# CONFIG
# ============================================================

ROOT = Path("/home/eng_megha/balldataset")

OUTPUT = ROOT / "annotation_visualization"

CASES = [
    "AI_train_caseB",
    "basic_train_caseA",
    "cc0_train_caseB",
    "match_train_caseA",
]

EXCLUDED = {
    "aug",
    "other",
    "volleyball",
}

# Number of images PER folder
SAMPLES_PER_FOLDER = 5

RANDOM_SEED = 42


# ============================================================
# IMAGE EXTENSIONS
# ============================================================

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


# ============================================================
# DRAW XYXY
# ============================================================

def draw_xyxy(image, rect):

    img = image.copy()

    draw = ImageDraw.Draw(img)

    x1, y1, x2, y2 = rect

    # Normalize in case coordinates are reversed
    left = min(x1, x2)
    right = max(x1, x2)

    top = min(y1, y2)
    bottom = max(y1, y2)

    draw.rectangle(
        [
            left,
            top,
            right,
            bottom,
        ],
        outline="red",
        width=4,
    )

    draw.text(
        (
            left,
            max(0, top - 20),
        ),
        "XYXY",
        fill="red",
    )

    return img


# ============================================================
# DRAW XYWH
# ============================================================

def draw_xywh(image, rect):

    img = image.copy()

    draw = ImageDraw.Draw(img)

    x, y, w, h = rect

    draw.rectangle(
        [
            x,
            y,
            x + w,
            y + h,
        ],
        outline="blue",
        width=4,
    )

    draw.text(
        (
            x,
            max(0, y - 20),
        ),
        "XYWH",
        fill="blue",
    )

    return img


# ============================================================
# DRAW BOTH
# ============================================================

def draw_both(image, rect):

    img = image.copy()

    draw = ImageDraw.Draw(img)

    # -------------------------
    # XYXY
    # -------------------------

    x1, y1, x2, y2 = rect

    left = min(x1, x2)
    right = max(x1, x2)

    top = min(y1, y2)
    bottom = max(y1, y2)

    draw.rectangle(
        [
            left,
            top,
            right,
            bottom,
        ],
        outline="red",
        width=4,
    )

    draw.text(
        (
            left,
            max(0, top - 20),
        ),
        "XYXY",
        fill="red",
    )

    # -------------------------
    # XYWH
    # -------------------------

    x, y, w, h = rect

    draw.rectangle(
        [
            x,
            y,
            x + w,
            y + h,
        ],
        outline="blue",
        width=4,
    )

    draw.text(
        (
            x,
            max(0, y - 20),
        ),
        "XYWH",
        fill="blue",
    )

    return img


# ============================================================
# PROCESS ONE IMAGE
# ============================================================

def process_image(
    image_path,
    json_path,
    output_xyxy,
    output_xywh,
    output_both,
):

    try:

        with open(
            json_path,
            "r",
            encoding="utf-8"
        ) as f:

            data = json.load(f)

    except Exception as e:

        print(
            f"[ERROR] JSON:"
            f" {json_path}"
        )

        return

    # --------------------------------------------------------
    # Open image
    # --------------------------------------------------------

    try:

        image = Image.open(
            image_path
        ).convert("RGB")

    except Exception as e:

        print(
            f"[ERROR] Image:"
            f" {image_path}"
        )

        return

    # --------------------------------------------------------
    # Get balls
    # --------------------------------------------------------

    balls = (
        data
        .get("data", {})
        .get("ball", [])
    )

    if not balls:

        return

    # --------------------------------------------------------
    # Draw every ball
    # --------------------------------------------------------

    xyxy_image = image.copy()
    xywh_image = image.copy()
    both_image = image.copy()

    xyxy_draw = ImageDraw.Draw(
        xyxy_image
    )

    xywh_draw = ImageDraw.Draw(
        xywh_image
    )

    both_draw = ImageDraw.Draw(
        both_image
    )

    for ball in balls:

        try:

            rect = (
                ball[
                    "entire"
                ][
                    "rect"
                ]
            )

        except Exception:

            continue

        if len(rect) != 4:
            continue

        x1, y1, x2, y2 = [
            float(v)
            for v in rect
        ]

        # ====================================================
        # XYXY
        # ====================================================

        left = min(x1, x2)
        right = max(x1, x2)

        top = min(y1, y2)
        bottom = max(y1, y2)

        xyxy_draw.rectangle(
            [
                left,
                top,
                right,
                bottom,
            ],
            outline="red",
            width=5,
        )

        xyxy_draw.text(
            (
                left,
                max(0, top - 20),
            ),
            "XYXY",
            fill="red",
        )

        # ====================================================
        # XYWH
        # ====================================================

        x = x1
        y = y1
        w = x2
        h = y2

        xywh_draw.rectangle(
            [
                x,
                y,
                x + w,
                y + h,
            ],
            outline="blue",
            width=5,
        )

        xywh_draw.text(
            (
                x,
                max(0, y - 20),
            ),
            "XYWH",
            fill="blue",
        )

        # ====================================================
        # BOTH
        # ====================================================

        both_draw.rectangle(
            [
                left,
                top,
                right,
                bottom,
            ],
            outline="red",
            width=5,
        )

        both_draw.rectangle(
            [
                x,
                y,
                x + w,
                y + h,
            ],
            outline="blue",
            width=5,
        )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    filename = image_path.name

    xyxy_image.save(
        output_xyxy / filename,
        quality=95,
    )

    xywh_image.save(
        output_xywh / filename,
        quality=95,
    )

    both_image.save(
        output_both / filename,
        quality=95,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    random.seed(
        RANDOM_SEED
    )

    # --------------------------------------------------------
    # Remove old visualization
    # --------------------------------------------------------

    if OUTPUT.exists():

        print(
            f"Removing old:"
            f" {OUTPUT}"
        )

        import shutil

        shutil.rmtree(
            OUTPUT
        )

    OUTPUT.mkdir(
        parents=True
    )

    total = 0

    # ========================================================
    # CASES
    # ========================================================

    for case in CASES:

        case_root = (
            ROOT /
            case
        )

        print(
            f"\n{'=' * 60}"
        )

        print(
            f"CASE: {case}"
        )

        print(
            f"{'=' * 60}"
        )

        # ----------------------------------------------------
        # Find imgs folders
        # ----------------------------------------------------

        for imgs_dir in sorted(
            case_root.rglob("imgs")
        ):

            if not imgs_dir.is_dir():
                continue

            # Ignore excluded folders
            if any(
                part.lower()
                in EXCLUDED
                for part in imgs_dir.parts
            ):
                continue

            labels_dir = (
                imgs_dir.parent /
                "labels"
            )

            if not labels_dir.exists():
                continue

            # Category name
            category = (
                imgs_dir
                .parent
                .name
            )

            # ------------------------------------------------
            # Find images
            # ------------------------------------------------

            images = []

            for image_path in (
                imgs_dir.iterdir()
            ):

                if not image_path.is_file():
                    continue

                if (
                    image_path.suffix.lower()
                    not in IMAGE_EXTENSIONS
                ):
                    continue

                json_path = (
                    labels_dir /
                    f"{image_path.stem}.json"
                )

                if json_path.exists():

                    images.append(
                        (
                            image_path,
                            json_path,
                        )
                    )

            if not images:
                continue

            # ------------------------------------------------
            # Random sample
            # ------------------------------------------------

            sample = random.sample(
                images,
                min(
                    SAMPLES_PER_FOLDER,
                    len(images),
                ),
            )

            print(
                f"\n{category}: "
                f"{len(images)} images"
            )

            print(
                f"Visualizing "
                f"{len(sample)}"
            )

            # ------------------------------------------------
            # Output folders
            # ------------------------------------------------

            xyxy_dir = (
                OUTPUT /
                "XYXY" /
                case /
                category
            )

            xywh_dir = (
                OUTPUT /
                "XYWH" /
                case /
                category
            )

            both_dir = (
                OUTPUT /
                "BOTH" /
                case /
                category
            )

            xyxy_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            xywh_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            both_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            # ------------------------------------------------
            # Process
            # ------------------------------------------------

            for image_path, json_path in sample:

                process_image(
                    image_path,
                    json_path,
                    xyxy_dir,
                    xywh_dir,
                    both_dir,
                )

                total += 1

    # ========================================================
    # DONE
    # ========================================================

    print(
        "\n" +
        "=" * 60
    )

    print(
        "DONE"
    )

    print(
        "=" * 60
    )

    print(
        f"Images visualized: {total}"
    )

    print(
        f"\nOutput:"
        f"\n{OUTPUT}"
    )

    print(
        "\nCompare:"
    )

    print(
        f"  {OUTPUT}/XYXY/"
    )

    print(
        f"  {OUTPUT}/XYWH/"
    )

    print(
        f"  {OUTPUT}/BOTH/"
    )


if __name__ == "__main__":
    main()
