from libreyolo import LibreYOLO

MODEL = "runs/train/picodet_pretrained_320/weights/best.pt"
DATA = "dataset.yaml"

model = LibreYOLO(MODEL)

print("Model:", MODEL)
print("Dataset:", DATA)
print("Running validation...")

results = model.val(
    data=DATA,
    imgsz=320,
    batch=32,
    workers=4,
    device="auto",
    split="val",
    verbose=True,
)

print("\n========== RESULTS ==========")

for k, v in results.items():
    print(f"{k}: {v}")
