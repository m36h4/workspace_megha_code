import libreyolo.models.picodet.trainer as picodet_trainer
import libreyolo.validation.preprocessors as val_preproc
from letterbox_picodet_rect import (
    LetterboxPICODETTrainTransform,
    LetterboxPICODETValPreprocessor,
)

picodet_trainer.PICODETTrainTransform = LetterboxPICODETTrainTransform
val_preproc.PICODETValPreprocessor = LetterboxPICODETValPreprocessor

from libreyolo.models.picodet.model import LibrePICODET

model = LibrePICODET(size="s", nb_classes=2)
results = model.train(
    data="dataset.yaml",
    epochs=2,           # smoke test first
    imgsz=(320, 480),   # height, width — no change needed
    batch=16,
    lr0=0.1,
    pretrained=False,
)
