import argparse
import os
from unittest.mock import patch

from nnunetv2.run.run_training import run_training_entry

from tuckercnn import TuckerContext
from tuckercnn.utils import read_yml


def main(run_cfg: dict) -> None:
    os.environ['nnUNet_preprocessed'] = run_cfg['nnUNet_preprocessed']
    os.environ['nnUNet_raw'] = run_cfg['nnUNet_raw']
    os.environ['nnUNet_results'] = run_cfg['nnUNet_results']

    run_args = [
        '',
        run_cfg['dataset_id'],
        '3d_fullres',
        '0',
        '-tr',
        'nnUNetTrainerNoMirroring',
    ]
    if run_cfg['pretrained_weights'] is not None:
        run_args.extend(['-pretrained_weights', run_cfg['pretrained_weights']])

    with patch('sys.argv', run_args), TuckerContext(config=run_cfg['tucker_config']):
        run_training_entry()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-c', '--config', type=str, default='configs/finetune_baseline.yml'
    )
    args = parser.parse_args()

    run_cfg = read_yml(args.config)
    main(run_cfg)
