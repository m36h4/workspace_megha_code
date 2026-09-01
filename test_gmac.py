import torch
from torchinfo import summary

# Apply the same patch first
import libreyolo.models.picodet.trainer as picodet_trainer
import libreyolo.validation.preprocessors as val_preproc

from letterbox_picodet_rect import (
    LetterboxPICODETTrainTransform,
    LetterboxPICODETValPreprocessor,
)

picodet_trainer.PICODETTrainTransform = LetterboxPICODETTrainTransform
val_preproc.PICODETValPreprocessor = LetterboxPICODETValPreprocessor

from libreyolo.models.picodet.model import LibrePICODET

model = LibrePICODET(
    size="s",
    nb_classes=2,
)

model.eval()

info = summary(
    model,
    input_size=(1, 3, 320, 480),
    col_names=("input_size", "output_size", "num_params", "mult_adds"),
    depth=4,
)

print(info)
print(f"\nGMACs: {info.total_mult_adds / 1e9:.4f}")
