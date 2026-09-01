python3 -c "
import libreyolo.models.picodet.trainer as picodet_trainer
import libreyolo.validation.preprocessors as val_preproc
from letterbox_picodet_rect import (
    LetterboxPICODETTrainTransform,
    LetterboxPICODETValPreprocessor,
)
picodet_trainer.PICODETTrainTransform = LetterboxPICODETTrainTransform
val_preproc.PICODETValPreprocessor = LetterboxPICODETValPreprocessor
print('patch applied OK')
print(picodet_trainer.PICODETTrainTransform)
"
