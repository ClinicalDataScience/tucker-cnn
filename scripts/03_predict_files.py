import sys
from pathlib import Path
from typing import Sequence

from totalsegmentator.python_api import totalsegmentator

from tuckercnn import TuckerContext
from tuckercnn.timer import Timer
from tuckercnn.utils import read_yml

Timer.verbose = False


def main(run_cfg: dict, f_paths: Sequence[str]) -> None:
    pred_dir = Path(run_cfg['out_root']) / run_cfg['out_id']
    pred_dir.mkdir(parents=True, exist_ok=True)

    with TuckerContext(run_cfg['tucker_config']):
        for f_path in f_paths:
            run_totalsegmentator(Path(f_path), pred_dir)


def run_totalsegmentator(f_path: Path, pred_dir: Path) -> None:
    subject_id = f_path.stem.split('.')[0]
    out_id = subject_id.replace('_0000', '')

    totalsegmentator(
        input=f_path,
        output=pred_dir / f'{out_id}',
        fast=run_cfg['fast_model'],
    )

    print(f'Subject {subject_id} done.')


if __name__ == "__main__":
    cfg_path = sys.argv[1]
    f_paths = sys.argv[2:]

    run_cfg = read_yml(cfg_path)
    main(run_cfg, f_paths)
