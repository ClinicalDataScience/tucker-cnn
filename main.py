import os

from totalsegmentator.python_api import totalsegmentator

from tuckercnn import TuckerContext
from tuckercnn.utils import read_nii, get_dice_score
from tuckercnn.timer import Timer

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
    'ckpt_path': 'checkpoints/tucker_test.pt',
    'save_model': False,
    'load_model': True,
}
# --------------------------------------------------------------------------------------


def main() -> None:
    import sys
    from tuckercnn.utils import read_yml
    cfg_path = sys.argv[1]
    run_cfg = read_yml(cfg_path)
    run_cfg['tucker_config']['save_model'] = True
    run_cfg['tucker_config']['load_model'] = False

    with TuckerContext(run_cfg['tucker_config']):
        totalsegmentator(input=IN_PATH, output=OUT_PATH, fast=run_cfg['fast_model'])

    # with TuckerContext(TUCKER_CONFIG):
    #     totalsegmentator(input=IN_PATH, output=OUT_PATH, fast=True)
    #
    # Timer.report()
    #
    # seg_true = read_nii(IN_LABEL)
    # seg_pred = read_nii(os.path.join(OUT_PATH, 'spleen.nii.gz'))
    # print(f'Dice Score: {get_dice_score(seg_true, seg_pred) :.3f}')


if __name__ == "__main__":
    main()
