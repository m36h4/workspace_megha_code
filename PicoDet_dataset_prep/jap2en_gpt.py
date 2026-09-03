import json
import shutil
import re
from pathlib import Path

# ============================================================
# CONFIGURATION
# ============================================================

# ------------------------------------------------------------
# ORIGINAL DATASET
# ------------------------------------------------------------

SOURCE_ROOT = Path(
    "/home/eng_megha/balldataset"
)


# ------------------------------------------------------------
# NEW DATASET
#
# This will have the same structure as SOURCE_ROOT.
# ------------------------------------------------------------

OUTPUT_ROOT = Path(
    "/home/eng_megha/balldataset_english_names"
)


# ------------------------------------------------------------
# IMAGE EXTENSIONS
# ------------------------------------------------------------

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
    ".tif",
    ".tiff",
}


# ============================================================
# JAPANESE DETECTION
# ============================================================

def contains_japanese(text):
    """
    Check whether a filename contains Japanese characters.

    Detects:
        Hiragana
        Katakana
        CJK / Kanji
    """

    japanese_pattern = re.compile(
        r"[\u3040-\u309F"   # Hiragana
        r"\u30A0-\u30FF"    # Katakana
        r"\u3400-\u4DBF"    # CJK Extension A
        r"\u4E00-\u9FFF"    # CJK Unified Ideographs / Kanji
        r"\uF900-\uFAFF]"    # CJK Compatibility Ideographs
    )

    return bool(
        japanese_pattern.search(text)
    )


# ============================================================
# JAPANESE -> ENGLISH / ROMAN CHARACTERS
# ============================================================

def create_converter():
    """
    Create the Japanese -> Roman/English converter.

    Requires:
        pip install pykakasi
    """

    try:
        import pykakasi

    except ImportError:

        print()
        print("=" * 70)
        print("ERROR: pykakasi is not installed")
        print("=" * 70)
        print()
        print(
            "Please install it with:"
        )
        print()
        print(
            "pip install pykakasi"
        )
        print()

        raise

    return pykakasi.kakasi()


KAKASI = None


def japanese_to_english(text):
    """
    Convert Japanese characters in a string to
    Latin/Roman characters.

    Non-Japanese characters are preserved.

    Example:

        test_サッカー_01
        ->
        test_sakkaa_01

    The exact Romanization depends on pykakasi.
    """

    global KAKASI

    if not contains_japanese(text):
        return text

    if KAKASI is None:
        KAKASI = create_converter()

    result = KAKASI.convert(text)

    converted = ""

    for item in result:
        converted += item["hepburn"]

    return converted


# ============================================================
# SAFE FILENAME
# ============================================================

def clean_filename(filename):
    """
    Convert Japanese characters in the filename.

    File extension is preserved separately.

    Example:

        ボール_001.jpg

    becomes something similar to:

        booru_001.jpg
    """

    path = Path(filename)

    stem = path.stem
    suffix = path.suffix

    new_stem = japanese_to_english(
        stem
    )

    return new_stem + suffix


# ============================================================
# PROCESS ONE IMAGE + JSON
# ============================================================

def process_image_and_json(
    image_path,
    label_path,
    output_imgs_dir,
    output_labels_dir,
):
    """
    Copy an image and its JSON annotation.

    If the image filename contains Japanese characters:

        image Japanese name
              |
              v
        English/Roman name

    The JSON gets the exact same new filename.

    Only JSON field changed:

        file_name

    Everything else remains unchanged.
    """

    original_image_name = (
        image_path.name
    )

    # --------------------------------------------------------
    # Generate new image filename
    # --------------------------------------------------------

    new_image_name = clean_filename(
        original_image_name
    )

    # --------------------------------------------------------
    # New JSON filename
    #
    # Same stem as the new image.
    # --------------------------------------------------------

    new_json_name = (
        Path(new_image_name).stem
        + ".json"
    )

    new_image_path = (
        output_imgs_dir /
        new_image_name
    )

    new_json_path = (
        output_labels_dir /
        new_json_name
    )

    # --------------------------------------------------------
    # Read JSON
    # --------------------------------------------------------

    with open(
        label_path,
        "r",
        encoding="utf-8"
    ) as f:

        annotation = json.load(f)

    # --------------------------------------------------------
    # Update ONLY file_name
    # --------------------------------------------------------

    if "file_name" in annotation:

        annotation["file_name"] = (
            new_image_name
        )

    else:

        # If file_name doesn't exist, add it.
        #
        # For your current JSON format this should
        # normally already exist.

        annotation["file_name"] = (
            new_image_name
        )

    # --------------------------------------------------------
    # Copy image
    # --------------------------------------------------------

    shutil.copy2(
        image_path,
        new_image_path
    )

    # --------------------------------------------------------
    # Save JSON
    #
    # Same JSON structure.
    # --------------------------------------------------------

    with open(
        new_json_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            annotation,
            f,
            indent=2,
            ensure_ascii=False
        )

    return (
        new_image_name,
        new_json_name
    )


# ============================================================
# PROCESS DATASET
# ============================================================

def process_dataset():
    """
    Process the entire original dataset.

    Directory structure is preserved.
    """

    print()
    print("=" * 75)
    print("JAPANESE FILENAME CLEANUP")
    print("=" * 75)

    print()
    print(
        f"Source:"
    )

    print(
        f"  {SOURCE_ROOT}"
    )

    print()
    print(
        f"Output:"
    )

    print(
        f"  {OUTPUT_ROOT}"
    )

    # --------------------------------------------------------
    # Check source
    # --------------------------------------------------------

    if not SOURCE_ROOT.exists():

        raise FileNotFoundError(
            f"\nSource dataset does not exist:\n"
            f"{SOURCE_ROOT}\n\n"
            f"Change SOURCE_ROOT in the script."
        )

    # --------------------------------------------------------
    # Create output root
    # --------------------------------------------------------

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Counters
    # --------------------------------------------------------

    total_images = 0
    total_json = 0

    renamed_images = 0
    unchanged_images = 0

    missing_json = 0
    invalid_json = 0

    copied_other_files = 0

    # --------------------------------------------------------
    # Walk through entire dataset
    # --------------------------------------------------------

    for current_dir in sorted(
        SOURCE_ROOT.rglob("*")
    ):

        if not current_dir.is_dir():
            continue

        # ----------------------------------------------------
        # Only process directories called imgs
        # ----------------------------------------------------

        if current_dir.name.lower() != "imgs":
            continue

        # ----------------------------------------------------
        # Corresponding labels directory
        # ----------------------------------------------------

        labels_dir = (
            current_dir.parent /
            "labels"
        )

        # ----------------------------------------------------
        # Output directories
        # ----------------------------------------------------

        relative_imgs_dir = (
            current_dir.relative_to(
                SOURCE_ROOT
            )
        )

        output_imgs_dir = (
            OUTPUT_ROOT /
            relative_imgs_dir
        )

        output_labels_dir = (
            output_imgs_dir.parent /
            "labels"
        )

        output_imgs_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        output_labels_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        print()
        print(
            "-" * 75
        )

        print(
            f"Processing:"
        )

        print(
            f"  {current_dir}"
        )

        print(
            f"Labels:"
        )

        print(
            f"  {labels_dir}"
        )

        # ----------------------------------------------------
        # Check labels directory
        # ----------------------------------------------------

        if not labels_dir.exists():

            print(
                "  [WARNING] labels directory "
                "does not exist."
            )

            # Copy images anyway
            # because user requested same structure.

            for file_path in sorted(
                current_dir.iterdir()
            ):

                if not file_path.is_file():
                    continue

                if (
                    file_path.suffix.lower()
                    not in IMAGE_EXTENSIONS
                ):
                    continue

                new_name = clean_filename(
                    file_path.name
                )

                shutil.copy2(
                    file_path,
                    output_imgs_dir /
                    new_name
                )

                total_images += 1

                if (
                    new_name !=
                    file_path.name
                ):

                    renamed_images += 1

                    print(
                        f"  RENAMED:"
                    )

                    print(
                        f"    {file_path.name}"
                    )

                    print(
                        f"    -> {new_name}"
                    )

                else:

                    unchanged_images += 1

            continue

        # ----------------------------------------------------
        # Process every image
        # ----------------------------------------------------

        for image_path in sorted(
            current_dir.iterdir()
        ):

            if not image_path.is_file():
                continue

            if (
                image_path.suffix.lower()
                not in IMAGE_EXTENSIONS
            ):
                continue

            total_images += 1

            # ------------------------------------------------
            # Find JSON using ORIGINAL image stem
            #
            # Example:
            #
            # image_日本.jpg
            # image_日本.json
            #
            # ------------------------------------------------

            label_path = (
                labels_dir /
                f"{image_path.stem}.json"
            )

            # ------------------------------------------------
            # Missing JSON
            # ------------------------------------------------

            if not label_path.exists():

                missing_json += 1

                print()
                print(
                    "  [MISSING JSON]"
                )

                print(
                    f"    Image: "
                    f"{image_path.name}"
                )

                print(
                    f"    Expected: "
                    f"{label_path.name}"
                )

                # ------------------------------------------------
                # Still copy the image
                # ------------------------------------------------

                new_image_name = (
                    clean_filename(
                        image_path.name
                    )
                )

                shutil.copy2(
                    image_path,
                    output_imgs_dir /
                    new_image_name
                )

                if (
                    new_image_name !=
                    image_path.name
                ):

                    renamed_images += 1

                else:

                    unchanged_images += 1

                continue

            total_json += 1

            # ------------------------------------------------
            # Read JSON
            # ------------------------------------------------

            try:

                with open(
                    label_path,
                    "r",
                    encoding="utf-8"
                ) as f:

                    json.load(f)

            except Exception as e:

                invalid_json += 1

                print()
                print(
                    "  [INVALID JSON]"
                )

                print(
                    f"    {label_path}"
                )

                print(
                    f"    {e}"
                )

                continue

            # ------------------------------------------------
            # Process image + JSON
            # ------------------------------------------------

            try:

                (
                    new_image_name,
                    new_json_name,
                ) = process_image_and_json(
                    image_path,
                    label_path,
                    output_imgs_dir,
                    output_labels_dir,
                )

            except Exception as e:

                print()
                print(
                    "  [ERROR]"
                )

                print(
                    f"    Image: "
                    f"{image_path}"
                )

                print(
                    f"    JSON: "
                    f"{label_path}"
                )

                print(
                    f"    {e}"
                )

                continue

            # ------------------------------------------------
            # Statistics
            # ------------------------------------------------

            if (
                new_image_name !=
                image_path.name
            ):

                renamed_images += 1

                print()
                print(
                    "  [RENAMED]"
                )

                print(
                    f"    Image:"
                )

                print(
                    f"      {image_path.name}"
                )

                print(
                    f"      -> "
                    f"{new_image_name}"
                )

                print(
                    f"    JSON:"
                )

                print(
                    f"      {label_path.name}"
                )

                print(
                    f"      -> "
                    f"{new_json_name}"
                )

            else:

                unchanged_images += 1

    # ========================================================
    # COPY NON-IMAGE / NON-JSON FILES
    # ========================================================

    print()
    print("=" * 75)
    print("COPYING OTHER FILES")
    print("=" * 75)

    """
    The main processing above handles imgs/labels.

    Here we preserve other files in the dataset as well.

    Examples:
        .txt
        .csv
        .yaml
        .md
        etc.

    Their names are also converted if they contain Japanese,
    because the requirement is to create an English-name copy
    while preserving the structure.
    """

    for source_file in SOURCE_ROOT.rglob("*"):

        if not source_file.is_file():
            continue

        # ----------------------------------------------------
        # Skip images and JSON files that were already handled
        # ----------------------------------------------------

        is_image = (
            source_file.suffix.lower()
            in IMAGE_EXTENSIONS
        )

        is_json = (
            source_file.suffix.lower()
            == ".json"
            and source_file.parent.name.lower()
            == "labels"
        )

        if (
            is_image
            and source_file.parent.name.lower()
            == "imgs"
        ):
            continue

        if is_json:
            continue

        # ----------------------------------------------------
        # Preserve relative directory structure
        # ----------------------------------------------------

        relative_path = (
            source_file.relative_to(
                SOURCE_ROOT
            )
        )

        output_parent = (
            OUTPUT_ROOT /
            relative_path.parent
        )

        output_parent.mkdir(
            parents=True,
            exist_ok=True
        )

        # ----------------------------------------------------
        # Clean filename
        # ----------------------------------------------------

        new_filename = clean_filename(
            source_file.name
        )

        destination = (
            output_parent /
            new_filename
        )

        # ----------------------------------------------------
        # Avoid copying files that were already generated
        # ----------------------------------------------------

        if destination.exists():
            continue

        shutil.copy2(
            source_file,
            destination
        )

        copied_other_files += 1

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print()
    print("=" * 75)
    print("FINISHED")
    print("=" * 75)

    print()
    print(
        f"Total images found       : "
        f"{total_images}"
    )

    print(
        f"Images renamed           : "
        f"{renamed_images}"
    )

    print(
        f"Images unchanged         : "
        f"{unchanged_images}"
    )

    print(
        f"JSON files processed     : "
        f"{total_json}"
    )

    print(
        f"Missing JSON files       : "
        f"{missing_json}"
    )

    print(
        f"Invalid JSON files       : "
        f"{invalid_json}"
    )

    print(
        f"Other files copied       : "
        f"{copied_other_files}"
    )

    print()
    print(
        "New dataset:"
    )

    print(
        f"  {OUTPUT_ROOT}"
    )

    print()
    print(
        "Original dataset was NOT modified."
    )

    print()
    print("=" * 75)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    process_dataset()
