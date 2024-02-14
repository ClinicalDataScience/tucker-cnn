from functools import partial
import multiprocessing as mp
from pathlib import Path

import pandas as pd
from totalsegmentator.map_to_binary import class_map

from tuckercnn.utils import read_nii, get_dice_score, get_surface_distance

# PARAMETERS
# --------------------------------------------------------------------------------------
IN_LABEL = '/data/core-rad/data/tucker/raw/000-tdata/labelsTs'
OUT_PATH = '<path_to_out>/output'

CSV_PATH = 'inference_test.csv'
NUM_WORKERS = mp.cpu_count()
# --------------------------------------------------------------------------------------


def main() -> None:
    label_dir = Path(IN_LABEL)
    pred_dir = Path(OUT_PATH)

    subject_ids = [entry.stem for entry in label_dir.iterdir()]

    func = partial(get_subject_metrics, label_dir=label_dir, pred_dir=pred_dir)
    pool = mp.Pool(NUM_WORKERS)
    results_list = pool.map_async(func, subject_ids)
    pool.close()
    pool.join()

    df = pd.DataFrame.from_records(results_list)
    df.to_csv(CSV_PATH)


def get_subject_metrics(subject_id: str, label_dir: Path, pred_dir: Path) -> dict:
    results = {}

    seg_true = read_nii(str(label_dir / f'{subject_id}.nii.gz'))

    label_dict = class_map['total']
    for idx, label_str in label_dict.items():
        seg_true = (seg_true == idx).astype(int)
        if seg_true.sum() == 0:
            ds = -1
            nsd = -1
        else:
            pred_file = pred_dir / f'{subject_id}' / label_str + '.nii.gz'
            seg_pred = read_nii(str(pred_file))

            ds = get_dice_score(seg_true, seg_pred)
            nsd = get_surface_distance(seg_true, seg_pred)

        results.update(
            {'subject_id': subject_id, 'label_str': label_str, 'ds': ds, 'nsd': nsd}
        )

    return results


if __name__ == '__main__':
    main()
