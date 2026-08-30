import json
import random
from pathlib import Path
from collections import Counter, defaultdict

from PIL import Image, ImageDraw, ImageFont


# ============================================================
# CONFIG
# ============================================================

ROOT = Path("/home/eng_megha/balldataset")

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

OUTPUT = ROOT / "format_proof"

SAMPLES_PER_TYPE = 20

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


# ============================================================
# VALIDITY
# ============================================================

def valid_xyxy(rect, W, H):

    x1, y1, x2, y2 = rect

    return (
        x1 >= 0
        and y1 >= 0
        and x2 > x1
        and y2 > y1
        and x2 <= W
        and y2 <= H
    )


def valid_xywh(rect, W, H):

    x, y, w, h = rect

    return (
        x >= 0
        and y >= 0
        and w > 0
        and h > 0
        and x + w <= W
        and y + h <= H
    )


# ============================================================
# DRAW BOX
# ============================================================

def draw_box(
    image,
    rect,
    fmt,
    color,
):

    img = image.copy()

    draw = ImageDraw.Draw(img)

    if fmt == "XYXY":

        x1, y1, x2, y2 = rect

    else:

        x, y, w, h = rect

        x1 = x
        y1 = y
        x2 = x + w
        y2 = y + h

    draw.rectangle(
        [
            x1,
            y1,
            x2,
            y2,
        ],
        outline=color,
        width=5,
    )

    draw.text(
        (
            x1,
            max(0, y1 - 25),
        ),
        fmt,
        fill=color,
    )

    return img


# ============================================================
# SIDE BY SIDE
# ============================================================

def make_proof_image(
    image,
    rect,
    filename,
    classification,
    W,
    H,
):

    # --------------------------------------------------------
    # XYXY image
    # --------------------------------------------------------

    left = draw_box(
        image,
        rect,
        "XYXY",
        "red",
    )

    # --------------------------------------------------------
    # XYWH image
    # --------------------------------------------------------

    right = draw_box(
        image,
        rect,
        "XYWH",
        "blue",
    )

    # --------------------------------------------------------
    # Same dimensions
    # --------------------------------------------------------

    gap = 20

    combined = Image.new(
        "RGB",
        (
            left.width * 2 + gap,
            left.height,
        ),
        "white",
    )

    combined.paste(
        left,
        (0, 0),
    )

    combined.paste(
        right,
        (
            left.width + gap,
            0,
        ),
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    filename = (
        f"{classification}_"
        f"{filename}"
    )

    output_file = (
        OUTPUT /
        classification /
        filename
    )

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    combined.save(
        output_file,
        quality=95,
    )

    return output_file


# ============================================================
# MAIN
# ============================================================

def main():

    random.seed(42)

    OUTPUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    overall = Counter()

    per_case = defaultdict(
        Counter
    )

    examples = defaultdict(list)

    total_images = 0
    total_boxes = 0

    # ========================================================
    # SCAN
    # ========================================================

    for case in CASES:

        print(
            f"\nScanning: {case}"
        )

        case_root = ROOT / case

        for imgs_dir in case_root.rglob("imgs"):

            if not imgs_dir.is_dir():
                continue

            # Ignore unwanted directories
            if any(
                p.lower() in EXCLUDED
                for p in imgs_dir.parts
            ):
                continue

            labels_dir = (
                imgs_dir.parent /
                "labels"
            )

            if not labels_dir.exists():
                continue

            for image_path in imgs_dir.iterdir():

                if (
                    image_path.suffix.lower()
                    not in IMAGE_EXTENSIONS
                ):
                    continue

                json_path = (
                    labels_dir /
                    f"{image_path.stem}.json"
                )

                if not json_path.exists():
                    continue

                total_images += 1

                try:

                    with open(
                        json_path,
                        "r",
                        encoding="utf-8",
                    ) as f:

                        data = json.load(f)

                except Exception:

                    overall["BAD_JSON"] += 1

                    continue

                # ------------------------------------------------
                # ACTUAL IMAGE SIZE
                # ------------------------------------------------

                try:

                    with Image.open(
                        image_path
                    ) as im:

                        W, H = im.size

                except Exception:

                    overall["BAD_IMAGE"] += 1

                    continue

                balls = (
                    data
                    .get("data", {})
                    .get("ball", [])
                )

                for ball_index, ball in enumerate(
                    balls
                ):

                    total_boxes += 1

                    try:

                        rect = (
                            ball[
                                "entire"
                            ][
                                "rect"
                            ]
                        )

                        rect = [
                            float(v)
                            for v in rect
                        ]

                    except Exception:

                        overall["BAD_RECT"] += 1

                        continue

                    if len(rect) != 4:

                        overall["BAD_RECT"] += 1

                        continue

                    xyxy = valid_xyxy(
                        rect,
                        W,
                        H,
                    )

                    xywh = valid_xywh(
                        rect,
                        W,
                        H,
                    )

                    # ------------------------------------------------
                    # CLASSIFICATION
                    # ------------------------------------------------

                    if xyxy and not xywh:

                        classification = (
                            "XYXY_ONLY"
                        )

                    elif xywh and not xyxy:

                        classification = (
                            "XYWH_ONLY"
                        )

                    elif xyxy and xywh:

                        classification = (
                            "BOTH_VALID"
                        )

                    else:

                        classification = (
                            "NEITHER_VALID"
                        )

                    overall[
                        classification
                    ] += 1

                    per_case[
                        case
                    ][
                        classification
                    ] += 1

                    # ------------------------------------------------
                    # Save examples
                    # ------------------------------------------------

                    if len(
                        examples[classification]
                    ) < SAMPLES_PER_TYPE:

                        examples[
                            classification
                        ].append(
                            {
                                "case": case,
                                "image": image_path,
                                "json": json_path,
                                "rect": rect,
                                "W": W,
                                "H": H,
                            }
                        )

    # ========================================================
    # RESULTS
    # ========================================================

    print(
        "\n"
        + "=" * 75
    )

    print(
        "ANNOTATION FORMAT ANALYSIS"
    )

    print(
        "=" * 75
    )

    print(
        f"Images scanned : {total_images}"
    )

    print(
        f"Boxes scanned  : {total_boxes}"
    )

    print()

    for key in [
        "XYXY_ONLY",
        "XYWH_ONLY",
        "BOTH_VALID",
        "NEITHER_VALID",
        "BAD_RECT",
        "BAD_JSON",
        "BAD_IMAGE",
    ]:

        print(
            f"{key:20s}: "
            f"{overall[key]}"
        )

    # ========================================================
    # PER CASE
    # ========================================================

    print(
        "\n"
        + "=" * 75
    )

    print(
        "PER CASE"
    )

    print(
        "=" * 75
    )

    for case in CASES:

        c = per_case[case]

        print(
            f"\n{case}"
        )

        print(
            f"  XYXY only    : "
            f"{c['XYXY_ONLY']}"
        )

        print(
            f"  XYWH only    : "
            f"{c['XYWH_ONLY']}"
        )

        print(
            f"  Both valid   : "
            f"{c['BOTH_VALID']}"
        )

        print(
            f"  Neither      : "
            f"{c['NEITHER_VALID']}"
        )

    # ========================================================
    # PROOF IMAGES
    # ========================================================

    print(
        "\n"
        + "=" * 75
    )

    print(
        "CREATING VISUAL PROOF"
    )

    print(
        "=" * 75
    )

    for classification, items in examples.items():

        print(
            f"\n{classification}: "
            f"{len(items)} examples"
        )

        for i, item in enumerate(items):

            try:

                image = Image.open(
                    item["image"]
                ).convert("RGB")

            except Exception:

                continue

            filename = (
                f"{item['case']}_"
                f"{item['image'].stem}_"
                f"{i}.jpg"
            )

            output = make_proof_image(
                image,
                item["rect"],
                filename,
                classification,
                item["W"],
                item["H"],
            )

            print(
                f"  {output}"
            )

    # ========================================================
    # TEXT REPORT
    # ========================================================

    report = (
        OUTPUT /
        "format_report.txt"
    )

    with open(
        report,
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            "ANNOTATION FORMAT REPORT\n"
        )

        f.write(
            "=" * 60 + "\n\n"
        )

        f.write(
            f"Images scanned: "
            f"{total_images}\n"
        )

        f.write(
            f"Boxes scanned: "
            f"{total_boxes}\n\n"
        )

        for key, value in overall.items():

            f.write(
                f"{key}: {value}\n"
            )

        f.write(
            "\n\nPER CASE\n"
        )

        for case in CASES:

            f.write(
                f"\n{case}\n"
            )

            for key, value in (
                per_case[case].items()
            ):

                f.write(
                    f"  {key}: {value}\n"
                )

        f.write(
            "\n\nEXAMPLES\n"
        )

        for classification, items in (
            examples.items()
        ):

            f.write(
                f"\n{classification}\n"
            )

            for item in items:

                f.write(
                    f"\n"
                    f"case: {item['case']}\n"
                    f"image: {item['image']}\n"
                    f"json: {item['json']}\n"
                    f"image_size: "
                    f"{item['W']}x{item['H']}\n"
                    f"rect: {item['rect']}\n"
                )

    print(
        "\n"
        + "=" * 75
    )

    print(
        "DONE"
    )

    print(
        "=" * 75
    )

    print(
        f"\nProof directory:"
        f"\n{OUTPUT}"
    )

    print(
        f"\nReport:"
        f"\n{report}"
    )


if __name__ == "__main__":
    main()
