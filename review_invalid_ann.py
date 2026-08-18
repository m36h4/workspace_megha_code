import os
import json
import shutil
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont


# ============================================================
# CONFIGURATION
# ============================================================

# CSV containing invalid annotations
CSV_FILE = r"C:\Users\q4761\Desktop\Data_analyzer\invalid_annotations.csv"

# Root directory of your original dataset
DATASET_ROOT = r"C:\Users\q4761\Desktop\workspace\BallDetection\balldataset"

# Output directory
OUTPUT_ROOT = r"C:\Users\q4761\Desktop\workspace\BallDetection\invalid_annotation_review"


# ============================================================
# OUTPUT DIRECTORIES
# ============================================================

ORIGINAL_ROOT = os.path.join(
    OUTPUT_ROOT,
    "original"
)

VISUALIZED_ROOT = os.path.join(
    OUTPUT_ROOT,
    "visualized"
)

REPORT_ROOT = os.path.join(
    OUTPUT_ROOT,
    "reports"
)

os.makedirs(ORIGINAL_ROOT, exist_ok=True)
os.makedirs(VISUALIZED_ROOT, exist_ok=True)
os.makedirs(REPORT_ROOT, exist_ok=True)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def clean_error_name(error):
    """
    Convert an error name from the CSV into a safe folder name.
    """

    if pd.isna(error):
        return "unknown_error"

    error = str(error).strip()

    error = error.replace("/", "_")
    error = error.replace("\\", "_")
    error = error.replace(" ", "_")

    return error


def normalize_relative_path(path):
    """
    Normalize Windows/Linux path separators.
    """

    path = str(path)

    path = path.replace("\\", os.sep)
    path = path.replace("/", os.sep)

    path = path.lstrip("\\/")

    return path


def find_image_from_json(json_path, json_data):
    """
    Dataset structure:

        .../<sport>/labels/image.json
        .../<sport>/imgs/image.jpg

    Uses file_name from JSON when available.
    """

    file_name = json_data.get("file_name")

    if file_name:
        image_name = Path(file_name).name
    else:
        image_name = json_path.stem

    json_parent = json_path.parent

    # Normal dataset structure:
    #
    # labels/
    # imgs/

    if json_parent.name.lower() == "labels":

        image_folder = json_parent.parent / "imgs"

    else:

        image_folder = json_parent.parent / "imgs"

    # --------------------------------------------------------
    # Try exact filename
    # --------------------------------------------------------

    candidate = image_folder / image_name

    if candidate.is_file():
        return candidate

    # --------------------------------------------------------
    # Try different image extensions
    # --------------------------------------------------------

    stem = Path(image_name).stem

    extensions = [
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".webp",
        ".JPG",
        ".JPEG",
        ".PNG",
    ]

    for ext in extensions:

        candidate = image_folder / (stem + ext)

        if candidate.is_file():
            return candidate

    # --------------------------------------------------------
    # Last resort: search entire dataset
    # --------------------------------------------------------

    for ext in extensions:

        matches = list(
            DATASET_ROOT.rglob(
                stem + ext
            )
        )

        if matches:
            return matches[0]

    return None


def get_font(size=18, bold=False):

    if bold:

        possible_fonts = [
            r"C:\Windows\Fonts\arialbd.ttf",
            r"C:\Windows\Fonts\Arial_Bold.ttf",
            r"C:\Windows\Fonts\calibrib.ttf",
        ]

    else:

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
                    size
                )

            except Exception:
                pass

    return ImageFont.load_default()


def draw_text_box(
    draw,
    position,
    text,
    font,
    text_color="red"
):
    """
    Draw readable text with a white background.
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
        outline=text_color,
        width=2,
    )

    draw.text(
        (x, y),
        text,
        fill=text_color,
        font=font,
    )


def safe_copy(source, destination):
    """
    Copy file while keeping the original untouched.
    """

    os.makedirs(
        os.path.dirname(destination),
        exist_ok=True
    )

    shutil.copy2(
        source,
        destination
    )


# ============================================================
# START
# ============================================================

print("=" * 80)
print("INVALID ANNOTATION REVIEW TOOL")
print("=" * 80)

print(
    f"\nCSV:"
    f"\n{CSV_FILE}"
)

print(
    f"\nDataset:"
    f"\n{DATASET_ROOT}"
)

print(
    f"\nOutput:"
    f"\n{OUTPUT_ROOT}"
)


# ============================================================
# CHECK INPUTS
# ============================================================

if not os.path.isfile(CSV_FILE):

    raise FileNotFoundError(
        f"\nCSV file not found:\n{CSV_FILE}"
    )


if not os.path.isdir(DATASET_ROOT):

    raise FileNotFoundError(
        f"\nDataset directory not found:\n{DATASET_ROOT}"
    )


# ============================================================
# READ CSV
# ============================================================

df = pd.read_csv(
    CSV_FILE
)

print(
    f"\nCSV rows found: {len(df):,}"
)


# ============================================================
# CHECK REQUIRED COLUMNS
# ============================================================

required_columns = [
    "file",
    "error"
]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:

    raise ValueError(
        "\nMissing required CSV columns: "
        + str(missing_columns)
        + "\nAvailable columns: "
        + str(list(df.columns))
    )


# ============================================================
# REPORT STORAGE
# ============================================================

summary_rows = []

bbox_rows = []

images_processed = 0
images_copied = 0
json_copied = 0

missing_json = 0
missing_image = 0

invalid_json = 0
image_open_errors = 0


# ============================================================
# PROCESS EVERY CSV ROW
# ============================================================

for row_number, row in df.iterrows():

    print("\n")
    print("-" * 80)

    print(
        f"[{row_number + 1}/{len(df)}]"
    )

    # --------------------------------------------------------
    # CSV VALUES
    # --------------------------------------------------------

    csv_file = str(
        row["file"]
    ).strip()

    csv_error = clean_error_name(
        row["error"]
    )

    print(
        f"Error : {csv_error}"
    )

    print(
        f"JSON  : {csv_file}"
    )

    # --------------------------------------------------------
    # NORMALIZE PATH
    # --------------------------------------------------------

    relative_path = normalize_relative_path(
        csv_file
    )

    json_path = os.path.join(
        DATASET_ROOT,
        relative_path
    )

    # --------------------------------------------------------
    # CHECK JSON
    # --------------------------------------------------------

    if not os.path.isfile(json_path):

        print(
            "[MISSING JSON]"
        )

        print(
            json_path
        )

        missing_json += 1

        summary_rows.append({

            "csv_row":
                row_number + 1,

            "json_file":
                csv_file,

            "error":
                csv_error,

            "status":
                "missing_json",

        })

        continue

    # ========================================================
    # LOAD JSON
    # ========================================================

    try:

        with open(
            json_path,
            "r",
            encoding="utf-8"
        ) as f:

            json_data = json.load(f)

    except Exception as e:

        print(
            f"[INVALID JSON] {e}"
        )

        invalid_json += 1

        summary_rows.append({

            "csv_row":
                row_number + 1,

            "json_file":
                csv_file,

            "error":
                csv_error,

            "status":
                "invalid_json",

        })

        continue

    # ========================================================
    # FIND IMAGE
    # ========================================================

    image_path = find_image_from_json(
        Path(json_path),
        json_data
    )

    if image_path is None:

        print(
            "[MISSING IMAGE]"
        )

        missing_image += 1

        summary_rows.append({

            "csv_row":
                row_number + 1,

            "json_file":
                csv_file,

            "error":
                csv_error,

            "status":
                "missing_image",

        })

        continue

    print(
        f"Image : {image_path}"
    )

    # ========================================================
    # COPY ORIGINAL JSON AND IMAGE
    # ========================================================

    original_error_folder = os.path.join(
        ORIGINAL_ROOT,
        csv_error
    )

    os.makedirs(
        original_error_folder,
        exist_ok=True
    )

    # --------------------------------------------------------
    # COPY JSON
    # --------------------------------------------------------

    destination_json = os.path.join(
        original_error_folder,
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
            f"[JSON COPY ERROR] {e}"
        )

    # --------------------------------------------------------
    # COPY IMAGE
    # --------------------------------------------------------

    destination_image = os.path.join(
        original_error_folder,
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
            f"[IMAGE COPY ERROR] {e}"
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
            f"[IMAGE OPEN ERROR] {e}"
        )

        image_open_errors += 1

        summary_rows.append({

            "csv_row":
                row_number + 1,

            "json_file":
                csv_file,

            "image_file":
                str(image_path),

            "error":
                csv_error,

            "status":
                "image_open_error",

        })

        continue

    # ========================================================
    # JSON DIMENSIONS
    # ========================================================

    declared_dimensions = json_data.get(
        "dimensions"
    )

    declared_height = None
    declared_width = None

    dimension_mismatch = False

    if (
        isinstance(
            declared_dimensions,
            list
        )
        and len(declared_dimensions) >= 2
    ):

        # Your JSON:
        #
        # "dimensions": [
        #     height,
        #     width
        # ]

        declared_height = float(
            declared_dimensions[0]
        )

        declared_width = float(
            declared_dimensions[1]
        )

        if (
            declared_width != actual_width
            or declared_height != actual_height
        ):

            dimension_mismatch = True

    # ========================================================
    # CREATE VISUALIZATION
    # ========================================================

    visual = image.copy()

    draw = ImageDraw.Draw(
        visual
    )

    normal_font = get_font(
        18,
        bold=False
    )

    large_font = get_font(
        24,
        bold=True
    )

    # --------------------------------------------------------
    # HEADER
    # --------------------------------------------------------

    header_lines = [

        f"ERROR: {csv_error}",

        f"Image: {image_path.name}",

        (
            f"Actual size: "
            f"{actual_width} x "
            f"{actual_height}"
        ),

    ]

    if (
        declared_width is not None
        and declared_height is not None
    ):

        header_lines.append(

            f"JSON size: "
            f"{int(declared_width)} x "
            f"{int(declared_height)}"

        )

    header_y = 10

    for line_number, line in enumerate(
        header_lines
    ):

        draw_text_box(

            draw,

            (
                10,
                header_y
            ),

            line,

            (
                large_font
                if line_number == 0
                else normal_font
            ),

            "red"

        )

        header_y += 35

    # ========================================================
    # PARSE ANNOTATIONS
    # ========================================================

    object_data = json_data.get(
        "data",
        {}
    )

    total_boxes = 0
    invalid_boxes = 0

    # ========================================================
    # EMPTY LABEL CHECK
    # ========================================================

    if not object_data:

        print(
            "[EMPTY LABEL DATA]"
        )

    # ========================================================
    # CLASSES
    # ========================================================

    for class_name, objects in object_data.items():

        if not isinstance(
            objects,
            list
        ):
            continue

        # ----------------------------------------------------
        # OBJECTS
        # ----------------------------------------------------

        for object_index, obj in enumerate(
            objects
        ):

            if not isinstance(
                obj,
                dict
            ):
                continue

            # ------------------------------------------------
            # ENTIRE
            # ------------------------------------------------

            entire = obj.get(
                "entire",
                {}
            )

            if not isinstance(
                entire,
                dict
            ):
                continue

            # ------------------------------------------------
            # RECT
            #
            # IMPORTANT:
            #
            # [x1, y1, x2, y2]
            # ------------------------------------------------

            rect = entire.get(
                "rect"
            )

            if not isinstance(
                rect,
                list
            ):
                continue

            if len(rect) != 4:

                print(
                    "[INVALID RECT]"
                    f" {rect}"
                )

                continue

            try:

                x1 = float(
                    rect[0]
                )

                y1 = float(
                    rect[1]
                )

                x2 = float(
                    rect[2]
                )

                y2 = float(
                    rect[3]
                )

            except Exception:

                print(
                    "[INVALID RECT VALUES]"
                )

                continue

            total_boxes += 1

            # ====================================================
            # VALIDATION
            # ====================================================

            box_errors = []

            # ----------------------------------------------------
            # NEGATIVE COORDINATES
            # ----------------------------------------------------

            if (
                x1 < 0
                or y1 < 0
                or x2 < 0
                or y2 < 0
            ):

                box_errors.append(
                    "negative_coordinate"
                )

            # ----------------------------------------------------
            # INVALID X
            # ----------------------------------------------------

            if x2 <= x1:

                box_errors.append(
                    "invalid_x_coordinates"
                )

            # ----------------------------------------------------
            # INVALID Y
            # ----------------------------------------------------

            if y2 <= y1:

                box_errors.append(
                    "invalid_y_coordinates"
                )

            # ----------------------------------------------------
            # IMAGE WIDTH
            # ----------------------------------------------------

            if (
                x1 > actual_width
                or x2 > actual_width
            ):

                box_errors.append(
                    "exceeds_image_width"
                )

            # ----------------------------------------------------
            # IMAGE HEIGHT
            # ----------------------------------------------------

            if (
                y1 > actual_height
                or y2 > actual_height
            ):

                box_errors.append(
                    "exceeds_image_height"
                )

            # ----------------------------------------------------
            # COMPLETELY OUTSIDE IMAGE
            # ----------------------------------------------------

            if (
                x2 <= 0
                or y2 <= 0
                or x1 >= actual_width
                or y1 >= actual_height
            ):

                box_errors.append(
                    "bbox_outside_image"
                )

            # ----------------------------------------------------
            # INVALID BOX?
            # ----------------------------------------------------

            is_invalid = (
                len(box_errors) > 0
            )

            if is_invalid:

                invalid_boxes += 1

            # ====================================================
            # DRAWING
            # ====================================================

            if is_invalid:

                box_color = "red"

            else:

                box_color = "lime"

            # ----------------------------------------------------
            # CLIP ONLY FOR VISUALIZATION
            #
            # Original coordinates are NOT changed.
            # ----------------------------------------------------

            draw_x1 = max(
                0,
                min(
                    int(x1),
                    actual_width - 1
                )
            )

            draw_y1 = max(
                0,
                min(
                    int(y1),
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

            # ----------------------------------------------------
            # DRAW RECTANGLE
            # ----------------------------------------------------

            if (
                draw_x2 > draw_x1
                and draw_y2 > draw_y1
            ):

                draw.rectangle(

                    [
                        draw_x1,
                        draw_y1,
                        draw_x2,
                        draw_y2,
                    ],

                    outline=box_color,

                    width=4

                )

            # ====================================================
            # BBOX LABEL
            # ====================================================

            bbox_label = (

                f"{class_name} | "

                f"x1={x1:g}, "

                f"y1={y1:g}, "

                f"x2={x2:g}, "

                f"y2={y2:g}"

            )

            if box_errors:

                bbox_label += (

                    " | "

                    + ", ".join(
                        box_errors
                    )

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

                normal_font,

                box_color

            )

            # ====================================================
            # BBOX CSV
            # ====================================================

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

                "x1":
                    x1,

                "y1":
                    y1,

                "x2":
                    x2,

                "y2":
                    y2,

                "image_width":
                    actual_width,

                "image_height":
                    actual_height,

                "negative_coordinate":
                    (
                        "negative_coordinate"
                        in box_errors
                    ),

                "exceeds_width":
                    (
                        "exceeds_image_width"
                        in box_errors
                    ),

                "exceeds_height":
                    (
                        "exceeds_image_height"
                        in box_errors
                    ),

                "bbox_outside_image":
                    (
                        "bbox_outside_image"
                        in box_errors
                    ),

                "bbox_valid":
                    not is_invalid,

                "calculated_errors":
                    ";".join(
                        box_errors
                    ),

            })

    # ========================================================
    # DIMENSION MISMATCH
    # ========================================================

    if dimension_mismatch:

        print(
            "[DIMENSION MISMATCH]"
        )

        print(
            f"JSON dimensions   : "
            f"{int(declared_width)} x "
            f"{int(declared_height)}"
        )

        print(
            f"Actual dimensions : "
            f"{actual_width} x "
            f"{actual_height}"
        )

    # ========================================================
    # SUMMARY ROW
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

    visualized_error_folder = os.path.join(
        VISUALIZED_ROOT,
        csv_error
    )

    os.makedirs(
        visualized_error_folder,
        exist_ok=True
    )

    visualized_path = os.path.join(
        visualized_error_folder,
        image_path.name
    )

    try:

        visual.save(
            visualized_path,
            quality=95
        )

        print(
            "[OK] Visualization:"
        )

        print(
            visualized_path
        )

    except Exception as e:

        print(
            f"[VISUALIZATION ERROR] {e}"
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
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# SAVE BBOX DETAILS CSV
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
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# ERROR TYPE SUMMARY
# ============================================================

error_summary = (
    df["error"]
    .fillna("unknown_error")
    .astype(str)
    .value_counts()
    .reset_index()
)

error_summary.columns = [
    "error",
    "count"
]

error_summary_csv = os.path.join(
    REPORT_ROOT,
    "error_type_summary.csv"
)

error_summary.to_csv(
    error_summary_csv,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# FINAL TERMINAL REPORT
# ============================================================

print("\n")
print("=" * 80)
print("PROCESSING COMPLETE")
print("=" * 80)

print(
    f"\nCSV rows processed       : {len(df):,}"
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
    f"Invalid JSON             : {invalid_json:,}"
)

print(
    f"Image open errors        : {image_open_errors:,}"
)

print(
    f"Bounding boxes analyzed  : {len(bbox_rows):,}"
)

print("\n")
print("=" * 80)
print("ERROR DISTRIBUTION")
print("=" * 80)

for _, error_row in error_summary.iterrows():

    print(
        f"{error_row['error']:<45}"
        f"{int(error_row['count']):>8,}"
    )


print("\n")
print("=" * 80)
print("OUTPUT FILES")
print("=" * 80)

print(
    f"\nOriginal copies:"
    f"\n{ORIGINAL_ROOT}"
)

print(
    f"\nVisualized images:"
    f"\n{VISUALIZED_ROOT}"
)

print(
    f"\nSummary CSV:"
    f"\n{summary_csv}"
)

print(
    f"\nBBox details CSV:"
    f"\n{bbox_csv}"
)

print(
    f"\nError summary CSV:"
    f"\n{error_summary_csv}"
)

print("\n")
print("=" * 80)
print("IMPORTANT: ORIGINAL DATASET WAS NOT MODIFIED")
print("=" * 80)
