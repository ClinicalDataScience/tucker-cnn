import torch
import os
import numpy as np
from skimage.transform import resize
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
from nnunetv2.inference.predict_from_raw_data import load_what_we_need
from monai.data import CacheDataset, DataLoader, Dataset, decollate_batch
from monai.inferers import sliding_window_inference
from tuckercnn.utils import read_nii, get_dice_score
from tuckercnn.timer import Timer
from nnunetv2.preprocessing.preprocessors.default_preprocessor import DefaultPreprocessor
import matplotlib.pyplot as plt
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



if __name__ == '__main__':
    TS_ROOT = os.path.join(os.environ['HOME'], '.totalsegmentator/nnunet/results')
    patch_size = (1, 112, 112, 128)
    model_path = (
        'Dataset297_TotalSegmentator_total_3mm_1559subj/'
        'nnUNetTrainer_4000epochs_NoMirroring__nnUNetPlans__3d_fullres'
    )
    path = os.path.join(TS_ROOT, model_path)
    data_d = [{"image": "/mnt/ssd/PycharmProjects/tucker-cnn/data/spleen/imagesTr/spleen_2.nii.gz",
               "label": "/mnt/ssd/PycharmProjects/tucker-cnn/data/spleen/labelsTr/spleen_2.nii.gz"}]

    mostly_useless_stuff = load_what_we_need(
        model_training_output_dir=path,
        use_folds=[0],
        checkpoint_name='checkpoint_final.pth',
    )
    network = mostly_useless_stuff[-2]
    params = mostly_useless_stuff[0][0]
    network.load_state_dict(params)

    val_keys = ["image", "label"]
    val_mode = ("bilinear", "nearest")
    test_keys = ["image"]
    test_mode = ("bilinear")

    keys = val_keys
    mode = val_mode
    pixdim = (3., 3., 3.)

    test_transforms = Compose(
        [
            LoadImaged(keys=keys),
            EnsureChannelFirstd(keys=keys),
            PreprocessNorm(
                keys=["image"],
                clip_values=(-1004.0, 1588.0),
                normalize_values=(-50.38697721419439, 503.39235619144),
            ),
            # CropForegroundd(keys=["image", "label"], source_key="image"),
            #Orientationd(keys=keys, axcodes="RAS"),
            Spacingd(keys=keys, pixdim=pixdim, mode=mode),  # pixdim=(1.5, 1.5, 1.5)
        ]
    )
    network = network.cuda()
    network.eval()
    check_ds = Dataset(data=data_d, transform=test_transforms)
    #check_loader = DataLoader(check_ds, batch_size=1, num_workers=4, collate_fn=decollate_batch)
    a = check_ds.__getitem__(0)
    inp = a["image"].permute(0,3,1,2)[None, ...].cuda()
    #seg_true = a["label"].permute(0,3,1,2).numpy()

    #inp = a["image"][None, ...].cuda()
    seg_true = a["label"].numpy()
    print(seg_true.min(), seg_true.max())

    print(inp.size())
    with torch.no_grad():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            with Timer() as t:
                res = sliding_window_inference(
                    inputs=inp,
                    roi_size=patch_size[1:],
                    sw_batch_size=1,
                    predictor=network,
                    overlap=0.25,
                    mode="gaussian",  # , "constant",
                )
                t.report()

    print(res.size())
    res = res.permute(0,1,3,4,2)
    pred = torch.softmax(res[0], dim=0)
    pred = pred.argmax(dim=0)

    seg_pred = pred.detach().cpu().numpy()
    print(seg_true.shape, seg_pred.shape)
    np.save("../notebooks/seg_true.npy", seg_true)
    np.save("../notebooks/seg_pred.npy", seg_pred)
    seg_spleen = np.where(seg_pred == 1, 1, 0)
    print(get_dice_score(seg_true[0], seg_spleen))

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Plot Image 1
    axes[0].imshow(np.amax(seg_true[0], axis=-1), cmap='gray')
    axes[0].set_title('Image 1')

    # Plot Image 2
    axes[1].imshow(np.amax(seg_pred, axis=-1), cmap='gray')
    axes[1].imshow(np.amax(seg_spleen, axis=-1), cmap='gray')
    axes[1].set_title('Image 2')

    # Plot the Absolute Difference
    #axes[2].imshow(difference, cmap='gray')
    #axes[2].set_title('Absolute Difference')

    # Save the plot as a PNG file
    plt.savefig('image_difference.png')
    #resizer = Spacing(pixdim=(3., 3., 3.), mode=("bilinear"))
    #reseg = resizer.inverse()

    #pp = DefaultPreprocessor()
    #plans_manager = mostly_useless_stuff[3]
    #dataset_json_file = mostly_useless_stuff[4]
    #data, _, properties = pp.run_case(["/mnt/ssd/PycharmProjects/tucker-cnn/data/spleen/imagesTr/spleen_2.nii.gz"],
    #                                  seg_file=None, plans_manager=plans_manager,
    #                                  configuration_manager=plans_manager.get_configuration('3d_fullres'),
    #                                  dataset_json=dataset_json_file)