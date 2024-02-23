from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
from pathlib import Path

import pandas as pd
from totalsegmentator.map_to_binary import class_map
from tqdm import tqdm

from tuckercnn.utils import read_nii, get_dice_score, get_surface_distance

# PARAMETERS
# --------------------------------------------------------------------------------------
IN_LABEL = '/data/core-rad/data/tucker/raw/000-tdata/labelsTs'
OUT_PATH = '/data/core-rad/data/tucker_predictions/test_run'

CSV_PATH = 'results/inference_test.csv'
NUM_WORKERS = 32
# --------------------------------------------------------------------------------------

def main() -> None:
    label_dir = Path(IN_LABEL)
    pred_dir = Path(OUT_PATH)

    subject_ids = [entry.stem.split('.')[0] for entry in label_dir.iterdir()]

    func = partial(get_subject_metrics, label_dir=label_dir, pred_dir=pred_dir)

    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {executor.submit(func, subject_id): subject_id for subject_id in
                   subject_ids}

        results_list = []
        with tqdm(total=len(subject_ids)) as pbar:
            for future in as_completed(futures):
                results_list.extend(future.result())
                pbar.update(1)

    df = pd.DataFrame.from_records(results_list)
    compute_mean_excluding_negatives(df)
    df.to_csv(CSV_PATH)


def get_subject_metrics(subject_id: str, label_dir: Path, pred_dir: Path) -> list:
    results = []

    seg_true = read_nii(str(label_dir / f'{subject_id}.nii.gz')).astype(int)

    label_dict = class_map['total']
    for idx, label_str in label_dict.items():
        seg_mask = (seg_true == idx).astype(int)
        if seg_mask.sum() == 0:
            ds = -1
            nsd = -1
        else:
            pred_file = pred_dir / f'{subject_id}' / (label_str + '.nii.gz')
            seg_pred = read_nii(str(pred_file))

            ds = get_dice_score(seg_mask, seg_pred)
            nsd = get_surface_distance(seg_mask, seg_pred)

        results.append(
            {'subject_id': subject_id, 'label_str': label_str, 'ds': ds, 'nsd': nsd}
        )

    tqdm.write(f'Subject {subject_id} done.')
    return results


def compute_mean_excluding_negatives(df):
    filtered_df = df[(df['ds'] != -1) & (df['nsd'] != -1)]
    means = filtered_df.groupby('label_str')[['ds', 'nsd']].mean()
    print(means)


if __name__ == '__main__':
    main()
