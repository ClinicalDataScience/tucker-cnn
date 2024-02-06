import torch
from monai.transforms import (  LoadImaged,
                                EnsureChannelFirstd,
                                Compose,
                                ScaleIntensityRanged,
                                CropForegroundd,
                                Orientationd,
                                Spacingd,
                                Spacing,
                                NormalizeIntensity,)

from monai.transforms.compose import MapTransform
from monai.data import CacheDataset, DataLoader, Dataset, decollate_batch

val_keys = ["image", "label"]
val_mode = ("bilinear", "nearest")
test_keys = ["image"]

test_transforms = Compose(
    [
        LoadImaged(keys=test_keys),
        EnsureChannelFirstd(keys=test_keys),
        PreprocessNorm(
            keys=test_keys,
            clip_values=(-1004.0, 1588.0),
            normalize_values=(-50.38697721419439, 503.39235619144),
        ),
        #CropForegroundd(keys=["image", "label"], source_key="image"),
        Orientationd(keys=test_keys, axcodes="RAS"),
        Spacingd(keys=test_keys, pixdim=(3., 3., 3.), mode=("bilinear")), #pixdim=(1.5, 1.5, 1.5)
    ]
)

class PreprocessNorm(MapTransform):
    """
    This transform class takes NNUNet's preprocessing method for reference.
    That code is in:
    https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunet/preprocessing/preprocessing.py

    """

    def __init__(
        self,
        keys,
        clip_values,
        normalize_values,
    ) -> None:
        super().__init__(keys)
        self.keys = keys
        self.low = clip_values[0]
        self.high = clip_values[1]
        self.mean = normalize_values[0]
        self.std = normalize_values[1]
        self.training = False
        self.normalize_intensity = NormalizeIntensity(nonzero=True, channel_wise=True)

    def __call__(self, data):
        # load data
        d = dict(data)
        image = d["image"]

        # clip image for CT dataset
        if self.low != 0 or self.high != 0:
            image = torch.clamp(image, self.low, self.high)
            image = (image - self.mean) / self.std
        else:
            image = self.normalize_intensity(image.copy())

        d["image"] = image
        return d