import os
import json
import shutil
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont


# ============================================================
# CONFIGURATION
# ============================================================

# Your invalid annotations CSV
CSV_FILE = r"C:\Users\q4761\Desktop\Data_analyzer\invalid_annotations.csv"

# Root of your dataset
DATASET_ROOT = r"C:\Users\q4761\Desktop\workspace\BallDetection\balldataset"

# Output directory
OUTPUT_ROOT = r"C:\Users\q4761\Desktop\workspace\BallDetection\invalid_annotation_review"


# ============================================================
# SETTINGS
# ============================================================

ORIGINAL_ROOT = os.path.join(OUTPUT_ROOT, "original")
VISUALIZED_ROOT = os.path.join(OUTPUT_ROOT, "visualized")
REPORT_ROOT = os.path.join(OUTPUT_ROOT, "reports")

os.makedirs(ORIGINAL_ROOT, exist_ok=True)
os.makedirs(VISUALIZED_ROOT, exist_ok=True)
os.makedirs(REPORT_ROOT, exist_ok=True)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def clean_error_name(error):
    """
    Convert CSV error names into safe folder names.
    """
    if pd.isna(error):
        return "unknown_error"

    error = str(error).strip()

    # Keep the error name readable
    error = error.replace("/", "_")
    error = error.replace("\\", "_")
    error = error.replace(" ", "_")

    return error


def find_image_from_json(json_path, json_data):
    """
    Your structure is:

        ...\labels\image.json
                    |
                    ---> ...\imgs\image.jpg

    We first use file_name from JSON.
    Then search common image extensions.
    """

    file_name = json_data.get("file_name")

    if file_name:
        image_name = Path(file_name).name
    else:
        image_name = json_path.stem

    # JSON is normally inside "labels"
    json_parent = json_path.parent

    if json_parent.name.lower() == "labels":
        image_folder = json_parent.parent / "imgs"
    else:
        image_folder = json_parent.parent / "imgs"

    possible_extensions = [
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".webp",
        ".JPG",
        ".JPEG",
        ".PNG",
    ]

    # If JSON specifies extension
    candidate = image_folder / image_name

    if candidate.is_file():
        return candidate

    # Search by stem with different extensions
    stem = Path(image_name).stem

    for ext in possible_extensions:
        candidate = image_folder / (stem + ext)

        if candidate.is_file():
            return candidate

    # If not found, recursively search dataset
    for ext in possible_extensions:
        matches = list(
            DATASET_ROOT.rglob(stem + ext)
        )

        if matches:
            return matches[0]

    return None


def get_font():
    """
    Try to get a readable font for annotations.
    """

    possible_fonts = [
        r"C:\Windows\Fonts\arial.ttf",
        r"C:\Windows\Fonts\Arial.ttf",
        r"C:\Windows\Fonts\calibri.ttf",
        r"C:\Windows\Fonts\Calibri.ttf",
    ]

    for font_path in possible_fonts:

        if os.path.isfile(font_path):

            try:
                return ImageFont.truetype(
                    font_path,
                    18
                )
            except:
                pass

    return ImageFont.load_default()


def get_large_font():
    """
    Larger font for error title.
    """

    possible_fonts = [
        r"C:\Windows\Fonts\arialbd.ttf",
        r"C:\Windows\Fonts\Arial_Bold.ttf",
        r"C:\Windows\Fonts\calibrib.ttf",
    ]

    for font_path in possible_fonts:

        if os.path.isfile(font_path):

            try:
                return ImageFont.truetype(
                    font_path,
                    24
                )
            except:
                pass

    return ImageFont.load_default()


def draw_text_box(draw, position, text, font, fill="red"):
    """
    Draw text with a background rectangle.
    """

    x, y = position

    bbox = draw.textbbox(
        (x, y),
        text,
        font=font
    )

    padding = 5

    draw.rectangle(
        [
            bbox[0] - padding,
            bbox[1] - padding,
            bbox[2] + padding,
            bbox[3] + padding,
        ],
        fill="white",
        outline=fill,
        width=2,
    )

    draw.text(
        (x, y),
        text,
        fill=fill,
        font=font,
    )


def safe_copy(src, dst):
    """
    Copy file without modifying the source.
    """

    os.makedirs(
        os.path.dirname(dst),
        exist_ok=True
    )

    shutil.copy2(
        src,
        dst
    )


# ============================================================
# START
# ============================================================

print("=" * 80)
print("INVALID ANNOTATION REVIEW TOOL")
print("=" * 80)

print(f"\nCSV:")
print(CSV_FILE)

print(f"\nDataset:")
print(DATASET_ROOT)

print(f"\nOutput:")
print(OUTPUT_ROOT)


# ============================================================
# CHECK INPUTS
# ============================================================

if not os.path.isfile(CSV_FILE):

    raise FileNotFoundError(
        f"\nCSV not found:\n{CSV_FILE}"
    )


if not os.path.isdir(DATASET_ROOT):

    raise FileNotFoundError(
        f"\nDataset root not found:\n{DATASET_ROOT}"
    )


# ============================================================
# READ CSV
# ============================================================

df = pd.read_csv(
    CSV_FILE
)

print(
    f"\nRows in CSV: {len(df):,}"
)


# ============================================================
# CHECK COLUMNS
# ============================================================

if "file" not in df.columns:

    raise ValueError(
        "\nCSV must contain a 'file' column."
    )


if "error" not in df.columns:

    raise ValueError(
        "\nCSV must contain an 'error' column."
    )


# ============================================================
# VARIABLES
# ============================================================

summary_rows = []
bbox_rows = []

images_processed = 0
images_copied = 0
json_copied = 0

missing_json = 0
missing_image = 0

json_errors = 0


# ============================================================
# PROCESS EACH CSV ROW
# ============================================================

for row_number, row in df.iterrows():

    csv_file = str(row["file"]).strip()
    csv_error = clean_error_name(row["error"])

    print("\n" + "-" * 80)

    print(
        f"[{row_number + 1}/{len(df)}]"
    )

    print(
        f"Error : {csv_error}"
    )

    print(
        f"File  : {csv_file}"
    )

    # --------------------------------------------------------
    # NORMALIZE PATH
    # --------------------------------------------------------

    relative_path = csv_file.replace(
        "\\",
        os.sep
    )

    relative_path = relative_path.replace(
        "/",
        os.sep
    )

    relative_path = relative_path.lstrip(
        "\\/"
    )

    # --------------------------------------------------------
    # JSON PATH
    # --------------------------------------------------------

    json_path = os.path.join(
        DATASET_ROOT,
        relative_path
    )

    if not os.path.isfile(json_path):

        print(
            "[WARNING] JSON not found:"
        )

        print(
            json_path
        )

        missing_json += 1

        summary_rows.append({
            "csv_row": row_number + 1,
            "json_file": csv_file,
            "error": csv_error,
            "status": "missing_json",
        })

        continue

    # --------------------------------------------------------
    # LOAD JSON
    # --------------------------------------------------------

    try:

        with open(
            json_path,
            "r",
            encoding="utf-8"
        ) as f:

            data = json.load(f)

    except Exception as e:

        print(
            f"[ERROR] Cannot read JSON: {e}"
        )

        json_errors += 1

        summary_rows.append({
            "csv_row": row_number + 1,
            "json_file": csv_file,
            "error": csv_error,
            "status": "invalid_json",
        })

        continue

    # --------------------------------------------------------
    # FIND IMAGE
    # --------------------------------------------------------

    image_path = find_image_from_json(
        Path(json_path),
        data
    )

    if image_path is None:

        print(
            "[WARNING] Corresponding image not found."
        )

        missing_image += 1

        summary_rows.append({
            "csv_row": row_number + 1,
            "json_file": csv_file,
            "error": csv_error,
            "status": "missing_image",
        })

        continue

    print(
        f"Image : {image_path}"
    )

    # ========================================================
    # COPY ORIGINAL JSON + IMAGE
    # ========================================================

    original_error_dir = os.path.join(
        ORIGINAL_ROOT,
        csv_error
    )

    os.makedirs(
        original_error_dir,
        exist_ok=True
    )

    # Copy JSON
    destination_json = os.path.join(
        original_error_dir,
        os.path.basename(json_path)
    )

    try:

        safe_copy(
            json_path,
            destination_json
        )

        json_copied += 1

    except Exception as e:

        print(
            f"[ERROR] JSON copy failed: {e}"
        )

    # Copy image
    destination_image = os.path.join(
        original_error_dir,
        image_path.name
    )

    try:

        safe_copy(
            str(image_path),
            destination_image
        )

        images_copied += 1

    except Exception as e:

        print(
            f"[ERROR] Image copy failed: {e}"
        )

    # ========================================================
    # OPEN IMAGE
    # ========================================================

    try:

        image = Image.open(
            image_path
        ).convert("RGB")

        actual_width, actual_height = image.size

    except Exception as e:

        print(
            f"[ERROR] Cannot open image: {e}"
        )

        summary_rows.append({
            "csv_row": row_number + 1,
            "json_file": csv_file,
            "error": csv_error,
            "status": "cannot_open_image",
        })

        continue

    # ========================================================
    # JSON DECLARED DIMENSIONS
    # ========================================================

    declared_dimensions = data.get(
        "dimensions"
    )

    declared_height = None
    declared_width = None

    dimension_mismatch = False

    if (
        isinstance(declared_dimensions, list)
        and len(declared_dimensions) >= 2
    ):

        declared_height = declared_dimensions[0]
        declared_width = declared_dimensions[1]

        if (
            declared_width != actual_width
            or declared_height != actual_height
        ):

            dimension_mismatch = True

    # ========================================================
    # VISUALIZATION
    # ========================================================

    visualized_error_dir = os.path.join(
        VISUALIZED_ROOT,
        csv_error
    )

    os.makedirs(
        visualized_error_dir,
        exist_ok=True
    )

    visual = image.copy()

    draw = ImageDraw.Draw(
        visual
    )

    font = get_font()
    large_font = get_large_font()

    # --------------------------------------------------------
    # Header
    # --------------------------------------------------------

    header_lines = [
        f"ERROR: {csv_error}",
        f"Image: {image_path.name}",
        f"Actual size: {actual_width} x {actual_height}",
    ]

    if (
        declared_width is not None
        and declared_height is not None
    ):

        header_lines.append(
            f"JSON size: {declared_width} x {declared_height}"
        )

    header_y = 10

    for line in header_lines:

        draw_text_box(
            draw,
            (10, header_y),
            line,
            large_font if line.startswith("ERROR") else font,
            fill="red"
        )

        header_y += 32

    # ========================================================
    # PARSE DATA
    # ========================================================

    object_data = data.get(
        "data",
        {}
    )

    total_boxes = 0
    invalid_boxes = 0

    # --------------------------------------------------------
    # Check empty labels
    # --------------------------------------------------------

    if not object_data:

        print(
            "[INFO] Empty labels / no annotation data."
        )

    # ========================================================
    # PROCESS CLASSES
    # ========================================================

    for class_name, objects in object_data.items():

        if not isinstance(objects, list):
            continue

        for object_index, obj in enumerate(objects):

            if not isinstance(obj, dict):
                continue

            # ------------------------------------------------
            # "entire"
            # ------------------------------------------------

            entire = obj.get(
                "entire",
                {}
            )

            if not isinstance(entire, dict):
                continue

            rect = entire.get(
                "rect"
            )

            if not isinstance(rect, list):
                continue

            if len(rect) != 4:
                continue

            try:

                x = float(rect[0])
                y = float(rect[1])
                w = float(rect[2])
                h = float(rect[3])

            except Exception:

                continue

            total_boxes += 1

            # ------------------------------------------------
            # x, y, w, h
            #
            # NOT x1,y1,x2,y2
            # ------------------------------------------------

            x2 = x + w
            y2 = y + h

            # ------------------------------------------------
            # VALIDATION
            # ------------------------------------------------

            errors_for_box = []

            if x < 0 or y < 0:

                errors_for_box.append(
                    "negative_coordinate"
                )

            if w <= 0 or h <= 0:

                errors_for_box.append(
                    "invalid_bbox_size"
                )

            if x2 > actual_width:

                errors_for_box.append(
                    "exceeds_image_width"
                )

            if y2 > actual_height:

                errors_for_box.append(
                    "exceeds_image_height"
                )

            if (
                x >= actual_width
                or y >= actual_height
            ):

                errors_for_box.append(
                    "bbox_outside_image"
                )

            # ------------------------------------------------
            # Is this box invalid?
            # ------------------------------------------------

            is_invalid = (
                len(errors_for_box) > 0
            )

            if is_invalid:

                invalid_boxes += 1

            # ------------------------------------------------
            # COLOR
            # ------------------------------------------------

            if is_invalid:

                outline_color = "red"
                text_color = "red"

            else:

                outline_color = "lime"
                text_color = "green"

            # ------------------------------------------------
            # DRAW RECTANGLE
            # ------------------------------------------------

            # Clamp only the visualization coordinates.
            #
            # The original coordinates are preserved in
            # the JSON and CSV.

            draw_x1 = max(
                0,
                min(
                    int(x),
                    actual_width - 1
                )
            )

            draw_y1 = max(
                0,
                min(
                    int(y),
                    actual_height - 1
                )
            )

            draw_x2 = max(
                0,
                min(
                    int(x2),
                    actual_width - 1
                )
            )

            draw_y2 = max(
                0,
                min(
                    int(y2),
                    actual_height - 1
                )
            )

            if draw_x2 > draw_x1 and draw_y2 > draw_y1:

                draw.rectangle(
                    [
                        draw_x1,
                        draw_y1,
                        draw_x2,
                        draw_y2,
                    ],
                    outline=outline_color,
                    width=4,
                )

            # ------------------------------------------------
            # LABEL
            # ------------------------------------------------

            bbox_label = (
                f"{class_name} | "
                f"x={x:g}, y={y:g}, "
                f"w={w:g}, h={h:g}"
            )

            if errors_for_box:

                bbox_label += (
                    " | "
                    + ", ".join(errors_for_box)
                )

            label_y = max(
                0,
                draw_y1 - 25
            )

            draw_text_box(
                draw,
                (
                    draw_x1,
                    label_y
                ),
                bbox_label,
                font,
                fill=text_color
            )

            # ------------------------------------------------
            # SAVE BBOX REPORT
            # ------------------------------------------------

            bbox_rows.append({

                "csv_row":
                    row_number + 1,

                "json_file":
                    csv_file,

                "image_file":
                    str(image_path),

                "csv_error":
                    csv_error,

                "class":
                    class_name,

                "object_index":
                    object_index,

                "x":
                    x,

                "y":
                    y,

                "width":
                    w,

                "height":
                    h,

                "x2":
                    x2,

                "y2":
                    y2,

                "image_width":
                    actual_width,

                "image_height":
                    actual_height,

                "negative_coordinate":
                    "negative_coordinate"
                    in errors_for_box,

                "exceeds_width":
                    "exceeds_image_width"
                    in errors_for_box,

                "exceeds_height":
                    "exceeds_image_height"
                    in errors_for_box,

                "bbox_outside_image":
                    "bbox_outside_image"
                    in errors_for_box,

                "bbox_valid":
                    not is_invalid,

                "calculated_errors":
                    ";".join(errors_for_box),
            })

    # ========================================================
    # DIMENSION MISMATCH REPORT
    # ========================================================

    if dimension_mismatch:

        print(
            "[WARNING] Declared dimensions mismatch:"
        )

        print(
            f"  JSON    : "
            f"{declared_width} x {declared_height}"
        )

        print(
            f"  Actual  : "
            f"{actual_width} x {actual_height}"
        )

    # ========================================================
    # SUMMARY
    # ========================================================

    summary_rows.append({

        "csv_row":
            row_number + 1,

        "json_file":
            csv_file,

        "image_file":
            str(image_path),

        "csv_error":
            csv_error,

        "actual_width":
            actual_width,

        "actual_height":
            actual_height,

        "declared_width":
            declared_width,

        "declared_height":
            declared_height,

        "dimension_mismatch":
            dimension_mismatch,

        "total_boxes":
            total_boxes,

        "invalid_boxes_found":
            invalid_boxes,

        "status":
            "processed",
    })

    # ========================================================
    # SAVE VISUALIZED IMAGE
    # ========================================================

    visualized_path = os.path.join(
        visualized_error_dir,
        image_path.name
    )

    try:

        visual.save(
            visualized_path,
            quality=95
        )

        print(
            f"[OK] Visualization saved:"
        )

        print(
            visualized_path
        )

    except Exception as e:

        print(
            f"[ERROR] Visualization failed: {e}"
        )

    images_processed += 1


# ============================================================
# SAVE SUMMARY CSV
# ============================================================

summary_csv = os.path.join(
    REPORT_ROOT,
    "invalid_annotation_summary.csv"
)

summary_df = pd.DataFrame(
    summary_rows
)

summary_df.to_csv(
    summary_csv,
    index=False
)


# ============================================================
# SAVE BBOX CSV
# ============================================================

bbox_csv = os.path.join(
    REPORT_ROOT,
    "bbox_details.csv"
)

bbox_df = pd.DataFrame(
    bbox_rows
)

bbox_df.to_csv(
    bbox_csv,
    index=False
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n")
print("=" * 80)
print("PROCESSING COMPLETE")
print("=" * 80)

print(
    f"\nCSV rows                 : {len(df):,}"
)

print(
    f"Images processed         : {images_processed:,}"
)

print(
    f"Images copied            : {images_copied:,}"
)

print(
    f"JSON files copied        : {json_copied:,}"
)

print(
    f"Missing JSON             : {missing_json:,}"
)

print(
    f"Missing images           : {missing_image:,}"
)

print(
    f"Invalid JSON files       : {json_errors:,}"
)

print(
    f"Bounding boxes analyzed  : {len(bbox_rows):,}"
)

print("\n")
print("OUTPUT:")
print(
    OUTPUT_ROOT
)

print("\n")
print("SUMMARY CSV:")
print(
    summary_csv
)

print("\n")
print("BBOX CSV:")
print(
    bbox_csv
)

print("\n")
print("=" * 80)
print("ORIGINAL DATASET WAS NOT MODIFIED")
print("=" * 80)
