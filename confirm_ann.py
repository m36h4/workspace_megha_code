import json
from pathlib import Path
from collections import Counter, defaultdict


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


# ============================================================
# HELPERS
# ============================================================

def is_excluded(path):

    return any(
        part.lower() in EXCLUDED
        for part in path.parts
    )


def valid_xyxy(rect, width, height):

    x1, y1, x2, y2 = rect

    return (
        x1 >= 0
        and y1 >= 0
        and x2 > x1
        and y2 > y1
        and x2 <= width
        and y2 <= height
    )


def valid_xywh(rect, width, height):

    x, y, w, h = rect

    return (
        x >= 0
        and y >= 0
        and w > 0
        and h > 0
        and x + w <= width
        and y + h <= height
    )


# ============================================================
# MAIN
# ============================================================

def main():

    overall = Counter()

    by_case = defaultdict(Counter)

    xyxy_only = []
    xywh_only = []
    both_valid = []
    neither_valid = []

    total_images = 0
    total_boxes = 0

    for case in CASES:

        case_root = ROOT / case

        print(
            f"\nScanning: {case}"
        )

        for imgs_dir in case_root.rglob("imgs"):

            if not imgs_dir.is_dir():
                continue

            if is_excluded(imgs_dir):
                continue

            labels_dir = (
                imgs_dir.parent /
                "labels"
            )

            if not labels_dir.exists():
                continue

            for image_path in imgs_dir.iterdir():

                if image_path.suffix.lower() not in {
                    ".jpg",
                    ".jpeg",
                    ".png",
                    ".bmp",
                    ".webp",
                }:
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
                        encoding="utf-8"
                    ) as f:

                        data = json.load(f)

                except Exception:

                    overall["bad_json"] += 1
                    continue

                # ------------------------------------------------
                # Dimensions from JSON
                # ------------------------------------------------

                dimensions = data.get(
                    "dimensions"
                )

                if (
                    not isinstance(dimensions, list)
                    or len(dimensions) != 2
                ):

                    overall["bad_dimensions"] += 1
                    continue

                width = float(
                    dimensions[0]
                )

                height = float(
                    dimensions[1]
                )

                # ------------------------------------------------
                # Balls
                # ------------------------------------------------

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
                            ball
                            ["entire"]
                            ["rect"]
                        )

                        rect = [
                            float(v)
                            for v in rect
                        ]

                    except Exception:

                        overall["bad_rect"] += 1
                        continue

                    if len(rect) != 4:

                        overall["bad_rect"] += 1
                        continue

                    xyxy = valid_xyxy(
                        rect,
                        width,
                        height
                    )

                    xywh = valid_xywh(
                        rect,
                        width,
                        height
                    )

                    # ------------------------------------------------
                    # Classify
                    # ------------------------------------------------

                    if xyxy and not xywh:

                        result = "XYXY_ONLY"

                        xyxy_only.append(
                            (
                                case,
                                str(json_path),
                                rect,
                                dimensions,
                            )
                        )

                    elif xywh and not xyxy:

                        result = "XYWH_ONLY"

                        xywh_only.append(
                            (
                                case,
                                str(json_path),
                                rect,
                                dimensions,
                            )
                        )

                    elif xyxy and xywh:

                        result = "BOTH_VALID"

                        both_valid.append(
                            (
                                case,
                                str(json_path),
                                rect,
                                dimensions,
                            )
                        )

                    else:

                        result = "NEITHER_VALID"

                        neither_valid.append(
                            (
                                case,
                                str(json_path),
                                rect,
                                dimensions,
                            )
                        )

                    overall[result] += 1
                    by_case[case][result] += 1

    # ========================================================
    # RESULTS
    # ========================================================

    print(
        "\n" +
        "=" * 75
    )

    print(
        "OVERALL RESULT"
    )

    print(
        "=" * 75
    )

    print(
        f"Images : {total_images}"
    )

    print(
        f"Boxes  : {total_boxes}"
    )

    print()

    for key in [
        "XYXY_ONLY",
        "XYWH_ONLY",
        "BOTH_VALID",
        "NEITHER_VALID",
        "bad_json",
        "bad_dimensions",
        "bad_rect",
    ]:

        print(
            f"{key:20s}: "
            f"{overall[key]}"
        )

    # ========================================================
    # PER CASE
    # ========================================================

    print(
        "\n" +
        "=" * 75
    )

    print(
        "RESULT BY CASE"
    )

    print(
        "=" * 75
    )

    for case in CASES:

        c = by_case[case]

        xyxy = c["XYXY_ONLY"]
        xywh = c["XYWH_ONLY"]
        both = c["BOTH_VALID"]
        neither = c["NEITHER_VALID"]

        print(
            f"\n{case}"
        )

        print(
            f"  XYXY only    : {xyxy}"
        )

        print(
            f"  XYWH only    : {xywh}"
        )

        print(
            f"  Both valid   : {both}"
        )

        print(
            f"  Neither      : {neither}"
        )

    # ========================================================
    # EXAMPLES
    # ========================================================

    print(
        "\n" +
        "=" * 75
    )

    print(
        "XYXY-ONLY EXAMPLES"
    )

    print(
        "=" * 75
    )

    for item in xyxy_only[:10]:

        case, path, rect, dims = item

        print(
            f"\n{case}"
        )

        print(
            f"  {path}"
        )

        print(
            f"  dimensions = {dims}"
        )

        print(
            f"  rect       = {rect}"
        )

    print(
        "\n" +
        "=" * 75
    )

    print(
        "XYWH-ONLY EXAMPLES"
    )

    print(
        "=" * 75
    )

    for item in xywh_only[:10]:

        case, path, rect, dims = item

        print(
            f"\n{case}"
        )

        print(
            f"  {path}"
        )

        print(
            f"  dimensions = {dims}"
        )

        print(
            f"  rect       = {rect}"
        )

    print(
        "\n" +
        "=" * 75
    )

    print(
        "BOTH-VALID EXAMPLES"
    )

    print(
        "=" * 75
    )

    for item in both_valid[:10]:

        case, path, rect, dims = item

        print(
            f"\n{case}"
        )

        print(
            f"  {path}"
        )

        print(
            f"  dimensions = {dims}"
        )

        print(
            f"  rect       = {rect}"
        )

    print(
        "\n" +
        "=" * 75
    )

    print(
        "NEITHER-VALID EXAMPLES"
    )

    print(
        "=" * 75
    )

    for item in neither_valid[:10]:

        case, path, rect, dims = item

        print(
            f"\n{case}"
        )

        print(
            f"  {path}"
        )

        print(
            f"  dimensions = {dims}"
        )

        print(
            f"  rect       = {rect}"
        )

    print(
        "\nDONE."
    )


if __name__ == "__main__":
    main()
