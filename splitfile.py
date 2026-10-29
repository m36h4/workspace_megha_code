python - <<'PY'
import os
import shutil

src = "/home/test_mohit/workspace_mohit/BallDetection/oct12/dataset"

files = sorted(
    f for f in os.listdir(src)
    if os.path.isfile(os.path.join(src, f))
)

n = len(files)
parts = 3

for i in range(parts):
    os.makedirs(os.path.join(src, f"part_{i+1}"), exist_ok=True)

for i, f in enumerate(files):
    dest = os.path.join(src, f"part_{i % parts + 1}", f)
    shutil.move(os.path.join(src, f), dest)

print(f"Total files: {n}")
for i in range(parts):
    folder = os.path.join(src, f"part_{i+1}")
    print(f"part_{i+1}: {len(os.listdir(folder))} files")
PY
find /home/test_mohit/workspace_mohit/BallDetection/oct12/dataset/part_{1,2,3} -maxdepth 1 -type f | wc -l
