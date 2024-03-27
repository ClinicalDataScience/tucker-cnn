import os

from totalsegmentator.python_api import totalsegmentator

from tuckercnn import TuckerContext
from tuckercnn.utils import read_nii, get_dice_score

# PARAMETERS
# --------------------------------------------------------------------------------------
IN_PATH = 'data/spleen/imagesTr/spleen_2.nii.gz'
IN_LABEL = 'data/spleen/labelsTr/spleen_2.nii.gz'
OUT_PATH = 'output'

TUCKER_CONFIG = {
    'tucker_args': {
        'rank_mode': 'relative',
        'rank_factor': 1 / 3,
        'rank_min': 16,
        'decompose': False,
        'verbose': True,
    },
    'apply_tucker': True,
    'inference_bs': 1,
    'ckpt_path': '',
    'save_model': False,
    'load_model': False,
}
# --------------------------------------------------------------------------------------


def main() -> None:
    with TuckerContext(TUCKER_CONFIG):
        totalsegmentator(input=IN_PATH, output=OUT_PATH, fast=True)

    seg_true = read_nii(IN_LABEL)
    seg_pred = read_nii(os.path.join(OUT_PATH, 'spleen.nii.gz'))
    print(f'Dice Score: {get_dice_score(seg_true, seg_pred) :.3f}')


if __name__ == "__main__":
    main()
