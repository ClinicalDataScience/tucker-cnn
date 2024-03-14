"""Create all checkpoints from config file grid."""
import os

from totalsegmentator.python_api import totalsegmentator

from tuckercnn import TuckerContext
from tuckercnn.monkey.config import MonkeyConfig
from tuckercnn.utils import read_yml

CONFIG_DIR = 'configs_grid'
CHECKPOINT_DIR = 'checkpoints'
IN_PATH = 'data/spleen/imagesTr/spleen_2.nii.gz'
IN_LABEL = 'data/spleen/labelsTr/spleen_2.nii.gz'
OUT_PATH = 'output'

def main() -> None:
    for cfg_file in os.listdir(CONFIG_DIR):
        cfg = read_yml(os.path.join(CONFIG_DIR, cfg_file))

        if not cfg['tucker_config']['apply_tucker']:
            continue

        TUCKER_CONFIG = {
            'tucker_args': cfg['tucker_config']['tucker_args'],
            'apply_tucker': True,
            'inference_bs': 1,
            'ckpt_path': os.path.join(CHECKPOINT_DIR, cfg_file.replace('.yml', '.pt')),
            'save_model': True,
            'load_model': False,
        }

        with TuckerContext(TUCKER_CONFIG):
            totalsegmentator(input=IN_PATH, output=OUT_PATH, fast=cfg['fast_model'])

        if hasattr(MonkeyConfig, 'orig_ckpt_path'):
            delattr(MonkeyConfig, 'orig_ckpt_path')


if __name__ == '__main__':
    main()