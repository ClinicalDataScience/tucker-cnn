import sys

import SimpleITK as sitk


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def read_nii(path):
    sitk_array = sitk.ReadImage(path, sitk.sitkFloat32)
    return sitk.GetArrayFromImage(sitk_array)


def get_dice_score(mask1, mask2):
    overlap = (mask1 * mask2).sum()
    sum = mask1.sum() + mask2.sum()
    dice_score = 2 * overlap / sum
    return dice_score
