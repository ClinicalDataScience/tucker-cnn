from functools import partial
import multiprocessing as mp
from pathlib import Path

from nnunetv2.inference import predict_from_raw_data
from nnunetv2.inference import sliding_window_prediction
from totalsegmentator import libs
from totalsegmentator.python_api import totalsegmentator
from tqdm import tqdm

from tuckercnn import monkey_patch
from tuckercnn.monkey_patch import MonkeyManager
from tuckercnn.timer import Timer

# PARAMETERS
# --------------------------------------------------------------------------------------
IN_ROOT = '/data/core-rad/data/tucker/raw/000-tdata/imagesTs'
OUT_ROOT = '/data/core-rad/data/tucker_predictions'
OUT_ID = 'test_run'

FAST_MODEL = True

MonkeyManager.apply_tucker = False
MonkeyManager.inference_bs = 1
MonkeyManager.tucker_args = {
    'rank_mode': 'relative',
    'rank_factor': 1 / 3,
    'rank_min': 16,
    'decompose': True,
    'verbose': True,
}
MonkeyManager.ckpt_path = ''
MonkeyManager.save_model = False
MonkeyManager.load_model = False
Timer.verbose = False

NUM_WORKERS = 8


# --------------------------------------------------------------------------------------


def main() -> None:
    libs.DummyFile = monkey_patch.DummyFile
    predict_from_raw_data.predict_from_raw_data = monkey_patch.predict_from_raw_data
    sliding_window_prediction.maybe_mirror_and_predict = (
        monkey_patch.maybe_mirror_and_predict
    )

    gt_dir = Path(IN_ROOT)
    pred_dir = Path(OUT_ROOT) / OUT_ID

    pred_dir.mkdir(parents=True, exist_ok=True)

    subject_ids = [subject.stem.split('.')[0] for subject in gt_dir.glob('*.nii.gz')]

    for subject_id in tqdm(subject_ids):
        run_totalsegmentator(subject_id, gt_dir, pred_dir)


def run_totalsegmentator(subject_id: str, gt_dir: Path, pred_dir: Path) -> None:
    out_id = subject_id.replace('_0000', '')

    totalsegmentator(
        input=gt_dir / f'{subject_id}.nii.gz',
        output=pred_dir / f'{out_id}',
        fast=FAST_MODEL,
    )

    print(f'Subject {subject_id} done.')


if __name__ == "__main__":
    main()
