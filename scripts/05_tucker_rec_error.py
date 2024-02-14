import os
import warnings

warnings.filterwarnings('ignore', category=DeprecationWarning)

import numpy as np
import pandas as pd
import tensorly
from tensorly import tucker_to_tensor
import torch
from tensorly.decomposition import partial_tucker
from torch.nn import Conv3d, ConvTranspose3d
from tqdm import trange
from nnunetv2.inference.predict_from_raw_data import load_what_we_need

from tuckercnn.utils import streamline_nnunet_architecture


TS_ROOT = os.path.join(os.environ['HOME'], '.totalsegmentator/nnunet/results')
FAST_MODEL = 0

tensorly.set_backend('numpy')


@torch.no_grad()
def main() -> None:
    if FAST_MODEL:
        model_path = (
            'Dataset297_TotalSegmentator_total_3mm_1559subj/'
            'nnUNetTrainer_4000epochs_NoMirroring__nnUNetPlans__3d_fullres'
        )
    else:
        model_path = (
            'Dataset291_TotalSegmentator_part1_organs_1559subj/'
            'nnUNetTrainerNoMirroring__nnUNetPlans__3d_fullres'
        )
    nnunet_path = os.path.join(TS_ROOT, model_path)

    # Load Network
    # ----------------------------------------------------------------------------------
    mostly_useless_stuff = load_what_we_need(
        model_training_output_dir=nnunet_path,
        use_folds=[0],
        checkpoint_name='checkpoint_final.pth',
    )
    network = mostly_useless_stuff[-2]

    params = mostly_useless_stuff[0][0]
    network.load_state_dict(params)

    network = streamline_nnunet_architecture(network)

    # Exec and eval Tucker decomposition
    # ----------------------------------------------------------------------------------
    metrics = {
        'name': [],
        'rank_0': [],
        'rank_1': [],
        'rec_error': [],
        'rel_error': [],
        'energy_error': [],
        'fast': [],
    }

    for name, module in network.named_modules():
        if not isinstance(module, (Conv3d, ConvTranspose3d)):
            continue

        weight = module.weight.data.numpy()

        print(name)

        rank_0_step_size = max(weight.shape[0] // 16, 1)
        rank_1_step_size = max(weight.shape[1] // 16, 1)

        for i in trange(0, weight.shape[0], rank_0_step_size):
            if i == 0:
                i = 1
            for j in trange(0, weight.shape[1], rank_1_step_size, leave=False):
                if j == 0:
                    j = 1
                rec_error, rel_error, energy_error = get_metrics(weight, ranks=(i, j))

                metrics['name'].append(name)
                metrics['rank_0'].append(i)
                metrics['rank_1'].append(j)
                metrics['rec_error'].append(rec_error)
                metrics['rel_error'].append(rel_error)
                metrics['energy_error'].append(energy_error)
                metrics['fast'].append(FAST_MODEL)

        df = pd.DataFrame.from_dict(metrics)
        df.to_csv('tucker_error_2.csv')


def get_metrics(
    weight: np.ndarray, ranks: tuple[int, int]
) -> tuple[float, float, float]:
    tucker, _ = partial_tucker(tensor=weight, modes=(0, 1), rank=ranks, init='svd')

    reconstruction = tucker_to_tensor(tucker, skip_factor=(3, 4))

    weight_norm = np.linalg.norm(weight)
    rec_error = np.linalg.norm(weight - reconstruction)
    rel_error = rec_error / weight_norm
    energy_error = rec_error**2 / weight_norm**2

    return rec_error, rel_error, energy_error


if __name__ == '__main__':
    main()
