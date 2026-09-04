from libreyolo.models.picodet.model import LibrePICODET

MODEL = "runs/train/picodet_pretrained_320/weights/best.pt"
DATA = "dataset.yaml"

model = LibrePICODET(
    size="s",
    nb_classes=1,
)

print("Loading:", MODEL)

# Load trained checkpoint
model.load(MODEL)

print("\nRunning validation...")
results = model.val(
    data=DATA,
    imgsz=320,
    batch=32,
    device="auto",
    workers=4,
)

print("\n========== VALIDATION RESULTS ==========")
print(results)
