from pathlib import Path
import numpy as np
import nibabel as nib
from tuckercnn.utils import read_nii, get_dice_score
from tuckercnn.utils import eprint
from tqdm import tqdm
from totalsegmentator.map_to_binary import class_map_5_parts

# PARAMETERS
# --------------------------------------------------------------------------------------
IN_LABEL = "/mnt/ssd/work/lmu/code/data/totaltest/labelsTs"
OUT_PATH = "/mnt/ssd/work/lmu/code/data/totaltest/output"
DS = "total"

# IN_LABEL = "/mnt/ssd/work/lmu/code/data/Task09_Spleen/labelsTr"
# OUT_PATH = "/mnt/ssd/work/lmu/code/data/Task09_Spleen/output"
# DS = "spleen"

# ------------------------------------------------------


def main() -> None:
    pred_dir = Path(OUT_PATH)
    label_dir = Path(IN_LABEL)
    subjects = [x.stem for x in pred_dir.iterdir()]

    dsc = []
    for subject in subjects:
        subject2 = subject
        if DS == "total":
            subject2 = subject.split("_")[0]

        try:
            seg_true = read_nii(label_dir / f"{subject2}.nii.gz")
            seg_pred = read_nii(pred_dir / f"{subject}" / 'spleen.nii.gz')
        except:
            # print("SITK error in loading the mask, fallback nibabel")
            seg_true = nib.load(label_dir / f"{subject2}.nii.gz").get_fdata()
            seg_pred = nib.load(pred_dir / f"{subject}" / 'spleen.nii.gz').get_fdata()

        seg_true = np.where(seg_true == 1, 1, 0)
        dc = get_dice_score(seg_true, seg_pred)
        tmp = None
        if seg_true.sum() != 0:
            dsc.append(dc)
        else:
            tmp = "Empty gt mask"
        eprint(subject, f' Dice Score: {dc:.3f} ', tmp)
    eprint("Mean Dice Score: ", np.mean(dsc))
    # break


if __name__ == "__main__":
    main()
