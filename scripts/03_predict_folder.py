import sys
from pathlib import Path
from typing import Sequence

from totalsegmentator.python_api import totalsegmentator

from tuckercnn import TuckerContext
from tuckercnn.timer import Timer

# PARAMETERS
# --------------------------------------------------------------------------------------
# IN_ROOT = '/data/core-rad/data/tucker/raw/000-tdata/imagesTs'
OUT_ROOT = '/data/core-rad/data/tucker_predictions'
OUT_ID = 'test_run'

FAST_MODEL = True

TUCKER_CONFIG = {
    'tucker_args': {
        'rank_mode': 'relative',
        'rank_factor': 1 / 3,
        'rank_min': 16,
        'decompose': True,
        'verbose': True,
    },
    'apply_tucker': True,
    'inference_bs': 1,
    'ckpt_path': '',
    'save_model': False,
    'load_model': False,
}

Timer.verbose = False
# --------------------------------------------------------------------------------------


def main(f_paths: Sequence[str]) -> None:
    pred_dir = Path(OUT_ROOT) / OUT_ID
    pred_dir.mkdir(parents=True, exist_ok=True)

    with TuckerContext(TUCKER_CONFIG):
        for f_path in f_paths:
            run_totalsegmentator(Path(f_path), pred_dir)


def run_totalsegmentator(f_path: Path, pred_dir: Path) -> None:
    subject_id = f_path.stem.split('.')[0]
    out_id = subject_id.replace('_0000', '')

    totalsegmentator(
        input=f_path,
        output=pred_dir / f'{out_id}',
        fast=FAST_MODEL,
    )

    print(f'Subject {subject_id} done.')


if __name__ == "__main__":
    f_paths = sys.argv[1:]
    main(f_paths)
