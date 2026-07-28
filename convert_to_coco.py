import os
import json
import shutil

# ============================================================
# CHANGE THESE PATHS
# ============================================================

SOURCE_ROOT = "/home/eng_megha/workspace_megha/balldataset"
OUTPUT_ROOT = "/home/eng_megha/PaddleDetection/dataset/ball_coco"

# ============================================================

TRAIN_GROUP = "basic_train_caseA"
VAL_GROUP = "basic_eval_caseA"

TRAIN_SPORTS = [
    "basketball",
    "rugby",
    "soccer",
    "volleyball",
]

VAL_SPORTS = [
    "americanfootball",
    "basketball",
    "rugby",
    "soccer",
]

CATEGORY_NAME = "ball"

os.makedirs(os.path.join(OUTPUT_ROOT, "train"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_ROOT, "val"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_ROOT, "annotations"), exist_ok=True)


def coco_template():
    return {
        "images": [],
        "annotations": [],
        "categories": [
            {
                "id": 1,
                "name": CATEGORY_NAME,
                "supercategory": CATEGORY_NAME,
            }
        ],
    }


train_json = coco_template()
val_json = coco_template()

image_id = 1
annotation_id = 1


def process_dataset(group_name, sports, split_name, coco_json):
    global image_id
    global annotation_id

    for sport in sports:
        img_dir = os.path.join(SOURCE_ROOT, group_name, sport, "imgs")
        label_dir = os.path.join(SOURCE_ROOT, group_name, sport, "labels")

        if not os.path.isdir(img_dir):
            print(f"Skipping missing folder: {img_dir}")
            continue

        count = 0

        for img_name in sorted(os.listdir(img_dir)):
            if not img_name.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            json_name = os.path.splitext(img_name)[0] + ".json"

            img_path = os.path.join(img_dir, img_name)
            json_path = os.path.join(label_dir, json_name)

            if not os.path.exists(json_path):
                print(f"Missing JSON: {json_path}")
                continue

            with open(json_path, "r") as f:
                ann = json.load(f)

            width, height = ann["dimensions"]

            new_filename = f"{sport}_{count:06d}.jpg"

            shutil.copy2(
                img_path,
                os.path.join(OUTPUT_ROOT, split_name, new_filename),
            )

            coco_json["images"].append({
                "id": image_id,
                "file_name": new_filename,
                "width": width,
                "height": height,
            })

            for cls_name, objects in ann["data"].items():
                if cls_name != "ball":
                    continue

                for obj in objects:
                    if "entire" not in obj:
                        continue
                    if "rect" not in obj["entire"]:
                        continue

                    x, y, w, h = obj["entire"]["rect"]

                    coco_json["annotations"].append({
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": 1,
                        "bbox": [float(x), float(y), float(w), float(h)],
                        "area": float(w * h),
                        "iscrowd": 0,
                    })

                    annotation_id += 1

            image_id += 1
            count += 1


process_dataset(TRAIN_GROUP, TRAIN_SPORTS, "train", train_json)
process_dataset(VAL_GROUP, VAL_SPORTS, "val", val_json)

with open(os.path.join(OUTPUT_ROOT, "annotations", "instances_train.json"), "w") as f:
    json.dump(train_json, f, indent=4)

with open(os.path.join(OUTPUT_ROOT, "annotations", "instances_val.json"), "w") as f:
    json.dump(val_json, f, indent=4)

print("=" * 60)
print("Conversion Finished")
print("=" * 60)
print("Training Images    :", len(train_json["images"]))
print("Training Boxes     :", len(train_json["annotations"]))
print("Validation Images  :", len(val_json["images"]))
print("Validation Boxes   :", len(val_json["annotations"]))
print("Output Folder      :", OUTPUT_ROOT)
