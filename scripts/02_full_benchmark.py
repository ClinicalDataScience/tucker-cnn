import csv
from itertools import product
import os
import warnings

import torch.cuda
from tqdm import tqdm

from tuckercnn.benchmark import exec_benchmark

warnings.filterwarnings("ignore")

OUT_FILE = 'benchmark_results.csv'
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
    'apply_tucker': True,
    'autocast': False,
    'compile': False,
    'eval_passes': 10,
    'ckpt_path': '.checkpoints/model.ckpt',
    'save_model': False,
    'load_model': False,
    '3mm': False,
}


def main() -> None:
    clear_file(OUT_FILE)

    rank_factors = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.9]
    rank_mins = [8]

    boolean_keys = ['apply_tucker', 'autocast', 'compile', '3mm']
    batch_size_max = 100

    it = tqdm(
        enumerate(product(rank_factors, rank_mins)),
        total=len(rank_factors) * len(rank_mins),
    )
    for i, (rank_factor, rank_min) in it:
        TUCKER_ARGS['rank_factor'] = rank_factor
        TUCKER_ARGS['rank_min'] = rank_min

        tqdm.write('--- Updating Tucker arguments.')

        for bs in range(1, batch_size_max + 1):
            runs_failed = []
            tqdm.write(f'--- Running Tucker configs for batch size {bs:3d}.')

            for b_dict in iterate_bool_combinations(boolean_keys):
                # Skip loop if full model was already handled
                if i > 0 and not b_dict['apply_tucker']:
                    continue

                try:
                    BENCHMARK_ARGS['batch_size'] = bs
                    BENCHMARK_ARGS.update(b_dict)
                    set_model_version(use_fast_model=BENCHMARK_ARGS['3mm'])

                    metrics = exec_benchmark(BENCHMARK_ARGS, TUCKER_ARGS, False)
                    tqdm.write('Config done.')

                    csv_row = BENCHMARK_ARGS | TUCKER_ARGS | metrics
                    write_dict_to_csv(OUT_FILE, csv_row)
                    runs_failed.append(False)

                except torch.cuda.OutOfMemoryError:
                    tqdm.write('Config failed. (OOM Error)')
                    runs_failed.append(True)

            if all(runs_failed):
                break


def set_model_version(use_fast_model: bool = True):
    if use_fast_model:
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

    BENCHMARK_ARGS['3mm'] = use_fast_model
    BENCHMARK_ARGS['patch_size'] = patch_size
    BENCHMARK_ARGS['nnunet_path'] = os.path.join(TS_ROOT, model_path)


def iterate_bool_combinations(keys: list[str]):
    for values in product([True, False], repeat=len(keys)):
        yield dict(zip(keys, values))


def clear_file(f_path: str):
    open(f_path, 'w').close()


def write_dict_to_csv(f_path: str, dict_data: dict):
    write_header = not os.path.isfile(f_path) or os.path.getsize(f_path) == 0

    with open(f_path, mode='a', newline='') as csvfile:
        fieldnames = dict_data.keys()
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(dict_data)


if __name__ == '__main__':
    main()
