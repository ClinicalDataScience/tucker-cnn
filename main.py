import os

from nnunetv2.inference import predict_from_raw_data
from nnunetv2.inference import sliding_window_prediction
from totalsegmentator import libs
from totalsegmentator.python_api import totalsegmentator

import monkey_patch
from utils import read_nii, get_dice_score, Timer

if __name__ == "__main__":
    IN_PATH = 'data/spleen/imagesTr/spleen_2.nii.gz'
    IN_LABEL = 'data/spleen/labelsTr/spleen_2.nii.gz'
    OUT_PATH = 'output'

    monkey_patch.APPLY_TUCKER = True
    monkey_patch.TUCKER_ARGS = {
        'rank_mode': 'relative',
        'rank_factor': 1 / 3,
        'rank_min': 16,
        'decompose': True,
        'verbose': True,
    }

    libs.DummyFile = monkey_patch.DummyFile
    predict_from_raw_data.predict_from_raw_data = monkey_patch.predict_from_raw_data
    sliding_window_prediction.maybe_mirror_and_predict = (
        monkey_patch.maybe_mirror_and_predict
    )

    totalsegmentator(input=IN_PATH, output=OUT_PATH, fast=True)

    Timer.report()

    seg_true = read_nii(IN_LABEL)
    seg_pred = read_nii(os.path.join(OUT_PATH, 'spleen.nii.gz'))
    print('DSC: ', get_dice_score(seg_true, seg_pred))
