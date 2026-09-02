python - <<'PY'
import json, os, re

img_dir = "dataset/balls/images"
json_file = "dataset/balls/annotations/instances_train.json"

with open(json_file, encoding="utf-8") as f:
    data = json.load(f)

used = set(os.listdir(img_dir))
rename = {}

for im in data["images"]:
    old = im["file_name"]
    base, ext = os.path.splitext(old)

    if not base.isascii():
        new = re.sub(r"[^A-Za-z0-9_.-]", "_", base) + ext

        # avoid filename collision
        i = 1
        while new in used and new != old:
            new = re.sub(r"[^A-Za-z0-9_.-]", "_", base) + f"_{i}" + ext
            i += 1

        os.rename(os.path.join(img_dir, old), os.path.join(img_dir, new))
        im["file_name"] = new
        used.add(new)
        rename[old] = new

with open(json_file, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False)

print(f"Renamed {len(rename)} images")
for a, b in rename.items():
    print(a, "->", b)
PY
