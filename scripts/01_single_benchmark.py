import os

from tuckercnn.benchmark import exec_benchmark

TS_ROOT = os.path.join(os.environ['HOME'], '.totalsegmentator/nnunet/results')


TUCKER_ARGS = {
    'rank_mode': 'relative',
    'rank_factor': 1 / 3,
    'rank_min': None,
    'decompose': False,
    'verbose': False,
}

BENCHMARK_ARGS = {
    'batch_size': 1,
    'device': 'cuda',
    'load_params': False,
    'apply_tucker': False,
    'autocast': False,
    'compile': False,
    'eval_passes': 10,
    'ckpt_path': '.checkpoints/model.ckpt',
    'save_model': False,
    'load_model': False,
    '3mm': False,
}

if __name__ == '__main__':
    if BENCHMARK_ARGS['3mm']:
        patch_size = (1, 112, 112, 128)
        model_path = (
            'Dataset297_TotalSegmentator_total_3mm_1559subj/'
            'nnUNetTrainer_4000epochs_NoMirroring__nnUNetPlans__3d_fullres'
        )
    else:
        patch_size = (1, 128, 128, 128)
        model_path = (
            'Dataset291_TotalSegmentator_part1_organs_1559subj/'
            'nnUNetTrainerNoMirroring__nnUNetPlans__3d_fullres'
        )

    BENCHMARK_ARGS['patch_size'] = patch_size
    BENCHMARK_ARGS['nnunet_path'] = os.path.join(TS_ROOT, model_path)
    results = exec_benchmark(BENCHMARK_ARGS, TUCKER_ARGS, True)
