"""
Mirror the ORIGINAL ball dataset into a new folder with the exact same
directory structure and JSON content -- the only thing that changes is
the filename of images/labels that contain Japanese characters, which
get transliterated (romanized) to English/ASCII.

Nothing else is touched:
  - Folder names stay the same.
  - JSON file CONTENT stays exactly the same (only the filename may change).
  - Filenames with no Japanese characters are copied unchanged.

Requires:
    pip install pykakasi --break-system-packages
"""

import csv
import re
import shutil
from pathlib import Path

import pykakasi


# ============================================================
# CONFIGURATION
# ============================================================

# The ORIGINAL dataset (same one used by the YOLO/COCO prep script)
SOURCE_ROOT = Path("/home/eng_megha/balldataset")

# New output folder -- same structure, renamed files only
OUTPUT_ROOT = Path("/home/eng_megha/balldataset_renamed")

# If True, nothing is copied -- just prints what WOULD happen.
DRY_RUN = False

# If True, overwrite OUTPUT_ROOT contents if they already exist.
OVERWRITE_EXISTING = True


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

    return bool(
        JAPANESE_PATTERN.search(text)
    )


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

    converted = "".join(
        item["hepburn"] for item in _kks.convert(text)
    )

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
# MAIN COPY + RENAME LOGIC
# ============================================================

def build_output_path(rel_dir, filename, used_names_by_dir):
    """
    Compute the final filename for a file, resolving collisions
    (two different original names romanizing to the same new name)
    by appending _1, _2, ... within the same output directory.
    """

    new_name = get_new_filename(filename)

    used_names = used_names_by_dir.setdefault(rel_dir, set())

    if new_name not in used_names:
        used_names.add(new_name)
        return new_name

    # Collision: append a numeric suffix before the extension
    stem = Path(new_name).stem
    suffix = Path(new_name).suffix

    counter = 1

    while True:
        candidate = f"{stem}_{counter}{suffix}"

        if candidate not in used_names:
            used_names.add(candidate)
            return candidate

        counter += 1


def mirror_dataset(source_root, output_root):
    """
    Walk source_root, replicate every directory into output_root,
    and copy every file -- renaming only the ones whose filename
    contains Japanese characters.
    """

    if not source_root.exists():
        raise FileNotFoundError(
            f"Source not found: {source_root}"
        )

    if output_root.exists() and not OVERWRITE_EXISTING:
        raise FileExistsError(
            f"Output already exists: {output_root}\n"
            f"Set OVERWRITE_EXISTING = True to write into it anyway."
        )

    output_root.mkdir(
        parents=True,
        exist_ok=True
    )

    used_names_by_dir = {}

    rows_for_log = []

    total_files = 0
    renamed_files = 0

    for dirpath, dirnames, filenames in _walk_sorted(source_root):

        rel_dir = Path(dirpath).relative_to(source_root)

        out_dir = output_root / rel_dir

        # ------------------------------------------------
        # Replicate directory (folder names never change)
        # ------------------------------------------------

        if not DRY_RUN:
            out_dir.mkdir(
                parents=True,
                exist_ok=True
            )

        for filename in filenames:

            total_files += 1

            src_file = Path(dirpath) / filename

            new_name = build_output_path(
                rel_dir,
                filename,
                used_names_by_dir
            )

            dst_file = out_dir / new_name

            was_renamed = (new_name != filename)

            if was_renamed:
                renamed_files += 1
                print(
                    f"[RENAME] {rel_dir / filename}  ->  "
                    f"{rel_dir / new_name}"
                )

            rows_for_log.append({
                "original_relative_path": str(rel_dir / filename),
                "new_relative_path": str(rel_dir / new_name),
                "renamed": was_renamed,
            })

            if not DRY_RUN:
                # JSON, images, anything else -> plain byte copy.
                # JSON content is NOT parsed/modified, so its
                # format/content stays 100% identical.
                shutil.copy2(
                    src_file,
                    dst_file
                )

    return total_files, renamed_files, rows_for_log


def _walk_sorted(root):
    """
    Like os.walk, but with deterministic (sorted) ordering, so results
    are reproducible across runs.
    """

    import os

    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        filenames.sort()
        yield dirpath, dirnames, filenames


def write_mapping_log(output_root, rows):
    """
    Write a CSV showing every file's original path -> new path,
    so renames are fully auditable.
    """

    log_path = output_root / "rename_mapping_log.csv"

    with open(
        log_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "original_relative_path",
                "new_relative_path",
                "renamed",
            ]
        )

        writer.writeheader()

        for row in rows:
            writer.writerow(row)

    print(
        f"\nMapping log saved to: {log_path}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n")
    print("=" * 70)
    print("ROMANIZE JAPANESE FILENAMES (structure-preserving copy)")
    print("=" * 70)

    print(f"\nSource: {SOURCE_ROOT}")
    print(f"Output: {OUTPUT_ROOT}")
    print(f"Dry run: {DRY_RUN}")

    total_files, renamed_files, rows = mirror_dataset(
        SOURCE_ROOT,
        OUTPUT_ROOT
    )

    print("\n")
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(f"Total files scanned : {total_files}")
    print(f"Files renamed       : {renamed_files}")
    print(f"Files unchanged     : {total_files - renamed_files}")

    if not DRY_RUN:
        write_mapping_log(OUTPUT_ROOT, rows)
    else:
        print(
            "\nDRY RUN -- nothing was written. "
            "Set DRY_RUN = False to actually copy the dataset."
        )


if __name__ == "__main__":
    main()
