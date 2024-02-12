import os
from pathlib import Path
import numpy as np
from nnunetv2.inference import predict_from_raw_data
from nnunetv2.inference import sliding_window_prediction
from totalsegmentator import libs
from totalsegmentator.python_api import totalsegmentator

from tuckercnn import monkey_patch
from tuckercnn.monkey_patch import MonkeyManager
from tuckercnn.utils import read_nii, get_dice_score
from tuckercnn.timer import Timer
from tuckercnn.utils import eprint
from tqdm import tqdm
from totalsegmentator.map_to_binary import class_map_5_parts

# PARAMETERS
# --------------------------------------------------------------------------------------
IN_PATH = "/mnt/ssd/work/lmu/code/data/totaltest/imagesTs"
IN_LABEL = "/mnt/ssd/work/lmu/code/data/totaltest/labelsTs"
OUT_PATH = "/mnt/ssd/work/lmu/code/data/totaltest/output"
DS = "total"

# IN_PATH = "/mnt/ssd/work/lmu/code/data/Task09_Spleen/imagesTr"
# IN_LABEL = "/mnt/ssd/work/lmu/code/data/Task09_Spleen/labelsTr"
# OUT_PATH = "/mnt/ssd/work/lmu/code/data/Task09_Spleen/output"
# DS = "spleen"

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
# --------------------------------------------------------------------------------------


def main() -> None:
    libs.DummyFile = monkey_patch.DummyFile
    predict_from_raw_data.predict_from_raw_data = monkey_patch.predict_from_raw_data
    sliding_window_prediction.maybe_mirror_and_predict = (
        monkey_patch.maybe_mirror_and_predict
    )

    gt_dir = Path(IN_PATH)
    pred_dir = Path(OUT_PATH)
    label_dir = Path(IN_LABEL)
    subjects = [x.stem.split(".")[0] for x in gt_dir.glob("*.nii.gz")]

    for subject in tqdm(subjects):
        try:
            totalsegmentator(
                input=gt_dir / f"{subject}.nii.gz",
                output=pred_dir / f"{subject}",
                fast=True,
            )
            Timer.report()
            try:
                subject2 = subject
                if DS == "total":
                    subject2 = subject.split("_")[0]
                seg_true = read_nii(label_dir / f"{subject2}.nii.gz")
                seg_true = np.where(seg_true == 1, 1, 0)
                seg_pred = read_nii(pred_dir / f"{subject}" / 'spleen.nii.gz')
                eprint(f'Dice Score: {get_dice_score(seg_true, seg_pred):.3f}')
            except:
                print("SITK error in loading the mask")
        except:
            pass
        # break


if __name__ == "__main__":
    main()
