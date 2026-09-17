"""
Mirror the ORIGINAL COCO-format ball dataset into a new folder with the
exact same directory structure, but:

  1. Any image (or other) filename containing Japanese characters is
     transliterated (romanized) to English/ASCII.
  2. The corresponding COCO annotation JSON for each split
     (instances_train.json / instances_val.json / instances_test.json)
     has its "images[*].file_name" fields updated so they still point
     to the correct (renamed) image files.
  3. Everything else in the JSON (ids, bboxes, categories,
     annotations, licenses, info, etc.) is left completely untouched.
  4. Filenames with no Japanese characters are copied unchanged, and
     their JSON entries are left as-is too.

Expected folder structure (matches the dataset this script targets):

    training_dataset/
        annotations/
            instances_train.json
            instances_val.json
            instances_test.json
        images/
            train/
            val/
            test/

Requires:
    pip install pykakasi --break-system-packages
"""

import json
import os
import re
import shutil
from pathlib import Path

import pykakasi


# ============================================================
# CONFIGURATION
# ============================================================

# The ORIGINAL dataset root (the folder that directly contains
# "annotations/" and "images/")
SOURCE_ROOT = Path("/home/eng_megha/balldataset/training_dataset")

# New output folder -- same structure, renamed files + synced JSON
OUTPUT_ROOT = Path("/home/eng_megha/balldataset_renamed/training_dataset")

# Splits to process. Must match both the images/<split> subfolder name
# and the annotations/instances_<split>.json file name.
SPLITS = ["train", "val", "test"]

# If True, nothing is copied/written -- just prints what WOULD happen.
DRY_RUN = False

# If True, overwrite OUTPUT_ROOT contents if they already exist.
OVERWRITE_EXISTING = True

# JSON indent used when re-saving annotation files (None = compact,
# matching json.dump's default single-line style is not typical, so
# 2 is used to keep it human-readable; content/values are unchanged).
JSON_INDENT = 2


# ============================================================
# JAPANESE CHARACTER DETECTION
# ============================================================

JAPANESE_PATTERN = re.compile(
    "["
    "\u3040-\u309F"   # Hiragana
    "\u30A0-\u30FF"   # Katakana
    "\u31F0-\u31FF"   # Katakana phonetic extensions
    "\u4E00-\u9FFF"   # CJK Unified Ideographs (kanji)
    "\u3400-\u4DBF"   # CJK extension A (rare kanji)
    "\uFF66-\uFF9F"   # Half-width katakana
    "\u3000-\u303F"   # CJK punctuation (full-width space, brackets, etc.)
    "]"
)


def contains_japanese(text):
    """
    Returns True if the text contains any Japanese character
    (hiragana, katakana, or kanji).
    """
    return bool(JAPANESE_PATTERN.search(text))


# ============================================================
# TRANSLITERATION
# ============================================================

_kks = pykakasi.kakasi()

# Characters that are unsafe/unwanted in filenames after romanization
_UNSAFE_CHARS = re.compile(r"[^A-Za-z0-9._\-]+")


def romanize(text):
    """
    Convert Japanese characters in `text` to romaji (Hepburn).
    Non-Japanese characters (letters, digits, punctuation) are
    preserved as-is.
    """
    converted = "".join(item["hepburn"] for item in _kks.convert(text))

    # Collapse any whitespace/odd separators pykakasi may introduce
    # into a single underscore, and strip anything not filename-safe.
    converted = converted.replace(" ", "_")
    converted = _UNSAFE_CHARS.sub("_", converted)

    # Avoid double/trailing underscores from the cleanup above
    converted = re.sub(r"_+", "_", converted).strip("_")

    return converted


def get_new_filename(original_name):
    """
    Given a filename (with extension), return a filename where any
    Japanese characters have been romanized. If there are no Japanese
    characters at all, the original name is returned unchanged
    (including original casing/formatting).
    """
    stem = Path(original_name).stem
    suffix = Path(original_name).suffix  # includes the leading dot

    if not contains_japanese(stem):
        return original_name

    new_stem = romanize(stem)

    if not new_stem:
        # Extremely unlikely (name was ALL Japanese punctuation, etc.)
        # -- fall back to a safe placeholder rather than an empty name.
        new_stem = "renamed"

    return f"{new_stem}{suffix}"


# ============================================================
# FILE RENAME / COPY LOGIC (per split image folder)
# ============================================================

def build_new_name(filename, used_names):
    """
    Compute the final filename for a file, resolving collisions
    (two different original names romanizing to the same new name)
    by appending _1, _2, ... within the same output directory.
    """
    new_name = get_new_filename(filename)

    if new_name not in used_names:
        used_names.add(new_name)
        return new_name

    stem = Path(new_name).stem
    suffix = Path(new_name).suffix
    counter = 1

    while True:
        candidate = f"{stem}_{counter}{suffix}"
        if candidate not in used_names:
            used_names.add(candidate)
            return candidate
        counter += 1


def process_image_folder(src_dir, dst_dir):
    """
    Copy every file from src_dir into dst_dir, romanizing filenames
    that contain Japanese characters.

    Returns a dict mapping ORIGINAL filename -> NEW filename for every
    file in this folder (identity mapping for unchanged names).
    """
    mapping = {}
    used_names = set()

    if not src_dir.exists():
        print(f"  [WARN] image folder not found, skipping: {src_dir}")
        return mapping

    if not DRY_RUN:
        dst_dir.mkdir(parents=True, exist_ok=True)

    for filename in sorted(os.listdir(src_dir)):
        src_file = src_dir / filename

        if not src_file.is_file():
            continue

        new_name = build_new_name(filename, used_names)
        mapping[filename] = new_name

        if new_name != filename:
            print(f"  [RENAME] {filename}  ->  {new_name}")

        if not DRY_RUN:
            shutil.copy2(src_file, dst_dir / new_name)

    return mapping


# ============================================================
# COCO JSON SYNC LOGIC
# ============================================================

def update_coco_file_names(json_path, filename_map, out_json_path):
    """
    Load a COCO annotation file, rewrite each images[*].file_name to
    use the renamed image filename (if it was renamed), and save it
    to out_json_path. Everything else in the JSON is left untouched.

    filename_map maps ORIGINAL basename -> NEW basename for the image
    files that live alongside this annotation file.
    """
    if not json_path.exists():
        print(f"  [WARN] annotation file not found, skipping: {json_path}")
        return 0

    with open(json_path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    updated_count = 0
    unmatched = []

    for image_entry in coco.get("images", []):
        original_file_name = image_entry.get("file_name", "")

        # COCO file_name is sometimes a bare filename, sometimes a
        # relative path (e.g. "train/xxx.jpg"). Only the basename is
        # what we renamed, so split it off and rebuild afterward.
        as_path = Path(original_file_name)
        base = as_path.name
        parent = as_path.parent  # "." if no subfolder was present

        if base in filename_map:
            new_base = filename_map[base]

            if new_base != base:
                new_file_name = (
                    new_base if str(parent) == "."
                    else str(parent / new_base)
                )
                image_entry["file_name"] = new_file_name
                updated_count += 1
        else:
            unmatched.append(original_file_name)

    if unmatched:
        print(
            f"  [WARN] {len(unmatched)} image(s) referenced in "
            f"{json_path.name} were not found in the image folder "
            f"(file_name left unchanged for these)."
        )

    if not DRY_RUN:
        out_json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_json_path, "w", encoding="utf-8") as f:
            json.dump(coco, f, ensure_ascii=False, indent=JSON_INDENT)

    return updated_count


# ============================================================
# COPY ANY EXTRA FILES IN annotations/ NOT COVERED BY SPLITS
# ============================================================

def copy_extra_annotation_files(src_annotations_dir, dst_annotations_dir, handled_names):
    """
    Copies any file inside annotations/ that isn't one of the
    instances_<split>.json files already processed (e.g. a README,
    a labels file, etc.), unchanged, renaming only if its filename
    contains Japanese characters.
    """
    if not src_annotations_dir.exists():
        return

    used_names = set(handled_names)

    if not DRY_RUN:
        dst_annotations_dir.mkdir(parents=True, exist_ok=True)

    for filename in sorted(os.listdir(src_annotations_dir)):
        if filename in handled_names:
            continue

        src_file = src_annotations_dir / filename
        if not src_file.is_file():
            continue

        new_name = build_new_name(filename, used_names)

        if new_name != filename:
            print(f"  [RENAME] annotations/{filename}  ->  annotations/{new_name}")

        if not DRY_RUN:
            shutil.copy2(src_file, dst_annotations_dir / new_name)


# ============================================================
# MAIN
# ============================================================

def main():
    print("\n" + "=" * 70)
    print("ROMANIZE JAPANESE FILENAMES + SYNC COCO ANNOTATIONS")
    print("=" * 70)

    print(f"\nSource: {SOURCE_ROOT}")
    print(f"Output: {OUTPUT_ROOT}")
    print(f"Dry run: {DRY_RUN}")

    if not SOURCE_ROOT.exists():
        raise FileNotFoundError(f"Source not found: {SOURCE_ROOT}")

    if OUTPUT_ROOT.exists() and not OVERWRITE_EXISTING:
        raise FileExistsError(
            f"Output already exists: {OUTPUT_ROOT}\n"
            f"Set OVERWRITE_EXISTING = True to write into it anyway."
        )

    if not DRY_RUN:
        OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    src_images_root = SOURCE_ROOT / "images"
    src_annotations_dir = SOURCE_ROOT / "annotations"
    dst_images_root = OUTPUT_ROOT / "images"
    dst_annotations_dir = OUTPUT_ROOT / "annotations"

    handled_annotation_filenames = set()
    total_renamed = 0
    total_updated_refs = 0

    for split in SPLITS:
        print(f"\n--- Split: {split} ---")

        src_split_dir = src_images_root / split
        dst_split_dir = dst_images_root / split

        filename_map = process_image_folder(src_split_dir, dst_split_dir)
        renamed_here = sum(1 for k, v in filename_map.items() if k != v)
        total_renamed += renamed_here

        json_name = f"instances_{split}.json"
        handled_annotation_filenames.add(json_name)

        src_json = src_annotations_dir / json_name
        dst_json = dst_annotations_dir / json_name

        updated_here = update_coco_file_names(src_json, filename_map, dst_json)
        total_updated_refs += updated_here

        print(
            f"  Images processed: {len(filename_map)} | "
            f"renamed: {renamed_here} | "
            f"annotation file_name entries updated: {updated_here}"
        )

    # Copy any other files sitting in annotations/ that weren't one of
    # the instances_<split>.json files (README, labelmap, etc.)
    copy_extra_annotation_files(
        src_annotations_dir,
        dst_annotations_dir,
        handled_annotation_filenames,
    )

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total image files renamed        : {total_renamed}")
    print(f"Total JSON file_name entries fixed: {total_updated_refs}")

    if DRY_RUN:
        print(
            "\nDRY RUN -- nothing was written. "
            "Set DRY_RUN = False to actually copy the dataset."
        )
    else:
        print(f"\nDone. Mirrored dataset written to: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
