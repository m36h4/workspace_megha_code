import os
import shutil
import pandas as pd
from pathlib import Path

# ============================================================
# CONFIGURATION
# ============================================================

# Your duplicate CSV
CSV_FILE = r"C:\Users\q4761\Desktop\duplicate_images.csv"

# Root of your original dataset
DATASET_ROOT = r"C:\Users\q4761\Desktop\workspace\BallDetection\balldataset"

# New folder where duplicate groups will be created
DUPLICATE_ROOT = r"C:\Users\q4761\Desktop\workspace\BallDetection\duplicate"

# ============================================================
# START
# ============================================================

print("=" * 70)
print("DUPLICATE IMAGE COPY TOOL")
print("=" * 70)

# Check paths
if not os.path.isfile(CSV_FILE):
    raise FileNotFoundError(f"CSV not found:\n{CSV_FILE}")

if not os.path.isdir(DATASET_ROOT):
    raise FileNotFoundError(f"Dataset root not found:\n{DATASET_ROOT}")

# Create duplicate folder
os.makedirs(DUPLICATE_ROOT, exist_ok=True)

# Read CSV
df = pd.read_csv(CSV_FILE)

print(f"\nCSV rows found: {len(df):,}")

# ------------------------------------------------------------
# Check required columns
# ------------------------------------------------------------

required_columns = ["file", "hash"]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"\nMissing required CSV columns: {missing_columns}\n"
        f"Available columns: {list(df.columns)}"
    )

# Remove rows with missing file/hash
df = df.dropna(subset=["file", "hash"])

# Convert to string
df["file"] = df["file"].astype(str)
df["hash"] = df["hash"].astype(str)

print(f"Valid rows: {len(df):,}")
print(f"Unique hashes: {df['hash'].nunique():,}")

# ============================================================
# STATISTICS
# ============================================================

images_copied = 0
json_copied = 0

missing_images = 0
missing_json = 0

errors = []

# Track duplicate groups
hash_counts = df["hash"].value_counts()

# ============================================================
# PROCESS EVERY ROW
# ============================================================

for index, row in df.iterrows():

    relative_file = row["file"]
    hash_value = row["hash"]

    # --------------------------------------------------------
    # Normalize Windows / Linux separators
    # --------------------------------------------------------

    relative_file = relative_file.replace("\\", os.sep)
    relative_file = relative_file.replace("/", os.sep)

    # Remove accidental leading separators
    relative_file = relative_file.lstrip("\\/")

    # --------------------------------------------------------
    # Source image
    # --------------------------------------------------------

    image_path = os.path.join(
        DATASET_ROOT,
        relative_file
    )

    # --------------------------------------------------------
    # Create hash folder
    # --------------------------------------------------------

    hash_folder = os.path.join(
        DUPLICATE_ROOT,
        hash_value
    )

    os.makedirs(hash_folder, exist_ok=True)

    # --------------------------------------------------------
    # COPY IMAGE
    # --------------------------------------------------------

    if os.path.isfile(image_path):

        destination_image = os.path.join(
            hash_folder,
            os.path.basename(image_path)
        )

        try:

            shutil.copy2(
                image_path,
                destination_image
            )

            images_copied += 1

        except Exception as e:

            errors.append(
                f"IMAGE COPY ERROR: {image_path} -> {e}"
            )

    else:

        print(f"[MISSING IMAGE] {image_path}")

        missing_images += 1

        continue

    # ========================================================
    # FIND CORRESPONDING JSON
    # ========================================================

    image_path_obj = Path(image_path)

    image_filename = image_path_obj.name
    image_stem = image_path_obj.stem

    # Expected structure:
    #
    # ...\class\imgs\image.jpg
    #                  ↓
    # ...\class\labels\image.json

    parent_folder = image_path_obj.parent

    if parent_folder.name.lower() == "imgs":

        labels_folder = parent_folder.parent / "labels"

        json_path = labels_folder / f"{image_stem}.json"

    else:

        # Fallback:
        # search in sibling labels directory

        labels_folder = parent_folder.parent / "labels"

        json_path = labels_folder / f"{image_stem}.json"

    # --------------------------------------------------------
    # COPY JSON
    # --------------------------------------------------------

    if json_path.is_file():

        destination_json = os.path.join(
            hash_folder,
            json_path.name
        )

        try:

            shutil.copy2(
                json_path,
                destination_json
            )

            json_copied += 1

        except Exception as e:

            errors.append(
                f"JSON COPY ERROR: {json_path} -> {e}"
            )

    else:

        print(
            f"[MISSING JSON] "
            f"{json_path}"
        )

        missing_json += 1


# ============================================================
# SUMMARY
# ============================================================

print("\n")
print("=" * 70)
print("COPY COMPLETE")
print("=" * 70)

print(f"\nCSV rows processed      : {len(df):,}")
print(f"Unique hash groups     : {df['hash'].nunique():,}")

print(f"\nImages copied           : {images_copied:,}")
print(f"JSON files copied      : {json_copied:,}")

print(f"\nMissing images         : {missing_images:,}")
print(f"Missing JSON files     : {missing_json:,}")

print(f"\nOutput directory:")
print(DUPLICATE_ROOT)

# ============================================================
# DUPLICATE GROUP SUMMARY
# ============================================================

print("\n")
print("=" * 70)
print("DUPLICATE GROUP SUMMARY")
print("=" * 70)

group_distribution = hash_counts.value_counts().sort_index()

for copies, number_of_groups in group_distribution.items():

    print(
        f"{number_of_groups:,} hash groups "
        f"contain {copies} identical image(s)"
    )

# ============================================================
# ERROR SUMMARY
# ============================================================

if errors:

    print("\n")
    print("=" * 70)
    print("ERRORS")
    print("=" * 70)

    for error in errors:
        print(error)

else:

    print("\nNo copy errors.")

print("\n")
print("=" * 70)
print("ORIGINAL DATASET WAS NOT MODIFIED")
print("=" * 70)
