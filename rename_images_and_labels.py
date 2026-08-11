from pathlib import Path
import re

# images/ and labels/ are in the same directory as this script
IMAGES_DIR = Path("images")
LABELS_DIR = Path("labels")

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp"
}


def clean_name(filename):
    """
    Remove Japanese and special characters from filenames.
    Keeps English letters, numbers, underscores, and hyphens.

    Examples:
        猫の画像①.jpg -> 01.jpg
        image 01!.jpg -> image_01.jpg
        car@001.png -> car001.png
    """
    path = Path(filename)

    # Replace spaces with underscores
    name = path.stem.replace(" ", "_")

    # Remove everything except English letters, numbers,
    # underscore, and hyphen
    name = re.sub(r"[^a-zA-Z0-9_-]", "", name)

    # Avoid empty filenames
    if not name:
        name = "image"

    return name + path.suffix.lower()


def rename_images_and_labels():
    if not IMAGES_DIR.exists():
        print("ERROR: images folder not found.")
        print(f"Expected folder: {IMAGES_DIR.resolve()}")
        return

    if not LABELS_DIR.exists():
        print("WARNING: labels folder not found.")
        print(f"Expected folder: {LABELS_DIR.resolve()}")

    for image_path in IMAGES_DIR.iterdir():
        if not image_path.is_file():
            continue

        if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        old_image_name = image_path.name
        old_stem = image_path.stem

        new_image_name = clean_name(old_image_name)
        new_stem = Path(new_image_name).stem

        new_image_path = IMAGES_DIR / new_image_name

        # Corresponding YOLO label
        old_label_path = LABELS_DIR / f"{old_stem}.txt"
        new_label_path = LABELS_DIR / f"{new_stem}.txt"

        # Nothing to change
        if old_image_name == new_image_name:
            continue

        # Don't overwrite an existing image
        if new_image_path.exists():
            print(f"SKIPPED: {old_image_name}")
            print(f"  Target already exists: {new_image_name}")
            continue

        # Rename image
        image_path.rename(new_image_path)

        # Rename corresponding label
        if old_label_path.exists():
            if new_label_path.exists():
                print(f"WARNING: Label already exists: {new_label_path.name}")
            else:
                old_label_path.rename(new_label_path)

            print(
                f"RENAMED: {old_image_name} -> {new_image_name}"
                f" | LABEL: {old_stem}.txt -> {new_stem}.txt"
            )
        else:
            print(f"RENAMED IMAGE: {old_image_name} -> {new_image_name}")
            print(f"  WARNING: No label found: {old_stem}.txt")


if __name__ == "__main__":
    rename_images_and_labels()
