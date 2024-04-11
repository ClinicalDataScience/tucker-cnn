import sys
import os
from pathlib import Path
import shutil
import json

import numpy as np
import nibabel as nib
import pandas as pd
import multiprocessing
from tqdm.auto import tqdm

from totalsegmentator.map_to_binary import class_map_5_parts, class_map

WORKER = 60
total_remap = { 'spleen': 1, 'kidney_right': 2, 'kidney_left': 3, 'gallbladder': 4, 'liver': 5, 'stomach': 6, 'pancreas': 7, 'adrenal_gland_right': 8, 'adrenal_gland_left': 9, 'lung_upper_lobe_left': 10, 'lung_lower_lobe_left': 11, 'lung_upper_lobe_right': 12, 'lung_middle_lobe_right': 13, 'lung_lower_lobe_right': 14, 'esophagus': 15, 'trachea': 16, 'thyroid_gland': 17, 'small_bowel': 18, 'duodenum': 19, 'colon': 20, 'urinary_bladder': 21, 'prostate': 22, 'kidney_cyst_left': 23, 'kidney_cyst_right': 24, 'sacrum': 25, 'vertebrae_S1': 26, 'vertebrae_L5': 27, 'vertebrae_L4': 28, 'vertebrae_L3': 29, 'vertebrae_L2': 30, 'vertebrae_L1': 31, 'vertebrae_T12': 32, 'vertebrae_T11': 33, 'vertebrae_T10': 34, 'vertebrae_T9': 35, 'vertebrae_T8': 36, 'vertebrae_T7': 37, 'vertebrae_T6': 38, 'vertebrae_T5': 39, 'vertebrae_T4': 40, 'vertebrae_T3': 41, 'vertebrae_T2': 42, 'vertebrae_T1': 43, 'vertebrae_C7': 44, 'vertebrae_C6': 45, 'vertebrae_C5': 46, 'vertebrae_C4': 47, 'vertebrae_C3': 48, 'vertebrae_C2': 49, 'vertebrae_C1': 50, 'heart': 51, 'aorta': 52, 'pulmonary_vein': 53, 'brachiocephalic_trunk': 54, 'subclavian_artery_right': 55, 'subclavian_artery_left': 56, 'common_carotid_artery_right': 57, 'common_carotid_artery_left': 58, 'brachiocephalic_vein_left': 59, 'brachiocephalic_vein_right': 60, 'atrial_appendage_left': 61, 'superior_vena_cava': 62, 'inferior_vena_cava': 63, 'portal_vein_and_splenic_vein': 64, 'iliac_artery_left': 65, 'iliac_artery_right': 66, 'iliac_vena_left': 67, 'iliac_vena_right': 68, 'humerus_left': 69, 'humerus_right': 70, 'scapula_left': 71, 'scapula_right': 72, 'clavicula_left': 73, 'clavicula_right': 74, 'femur_left': 75, 'femur_right': 76, 'hip_left': 77, 'hip_right': 78, 'spinal_cord': 79, 'gluteus_maximus_left': 80, 'gluteus_maximus_right': 81, 'gluteus_medius_left': 82, 'gluteus_medius_right': 83, 'gluteus_minimus_left': 84, 'gluteus_minimus_right': 85, 'autochthon_left': 86, 'autochthon_right': 87, 'iliopsoas_left': 88, 'iliopsoas_right': 89, 'brain': 90, 'skull': 91, 'rib_left_1': 92, 'rib_left_2': 93, 'rib_left_3': 94, 'rib_left_4': 95, 'rib_left_5': 96, 'rib_left_6': 97, 'rib_left_7': 98, 'rib_left_8': 99, 'rib_left_9': 100, 'rib_left_10': 101, 'rib_left_11': 102, 'rib_left_12': 103, 'rib_right_1': 104, 'rib_right_2': 105, 'rib_right_3': 106, 'rib_right_4': 107, 'rib_right_5': 108, 'rib_right_6': 109, 'rib_right_7': 110, 'rib_right_8': 111, 'rib_right_9': 112, 'rib_right_10': 113, 'rib_right_11': 114, 'rib_right_12': 115, 'sternum': 116, 'costal_cartilages': 117}

def generate_json_from_dir_v2(foldername, subjects_train, subjects_val, labels):
    print("Creating dataset.json...")
    print("nnUNet env vars: ", os.environ['nnUNet_raw'], os.environ['nnUNet_preprocessed'])
    out_base = Path(os.environ['nnUNet_raw']) / foldername

    json_dict = {}
    json_dict['name'] = "TotalSegmentator"
    json_dict['description'] = "Segmentation of TotalSegmentator classes"
    json_dict['reference'] = "https://zenodo.org/record/6802614"
    json_dict['licence'] = "Apache 2.0"
    json_dict['release'] = "2.0"
    json_dict['channel_names'] = {"0": "CT"}
    json_dict['labels'] = {val:idx for idx,val in enumerate(["background",] + list(labels))}
    json_dict['numTraining'] = len(subjects_train + subjects_val)
    json_dict['file_ending'] = '.nii.gz'
    json_dict['overwrite_image_reader_writer'] = 'NibabelIOWithReorient'

    json.dump(json_dict, open(out_base / "dataset.json", "w"), sort_keys=False, indent=4)

    print("Creating split_final.json...")
    output_folder_pkl = Path(os.environ['nnUNet_preprocessed']) / foldername
    output_folder_pkl.mkdir(exist_ok=True)

    splits = []
    splits.append({
        "train": subjects_train,
        "val": subjects_val
    })

    print(f"nr of folds: {len(splits)}")
    print(f"nr train subjects (fold 0): {len(splits[0]['train'])}")
    print(f"nr val subjects (fold 0): {len(splits[0]['val'])}")

    json.dump(splits, open(output_folder_pkl / "splits_final.json", "w"), sort_keys=False, indent=4)


def combine_labels(ref_img, file_out, masks):
    ref_img = nib.load(ref_img)
    combined = np.zeros(ref_img.shape).astype(np.uint8)
    for idx, arg in enumerate(masks):
        file_in = Path(arg)
        if file_in.exists():
            img = nib.load(file_in)
            combined[img.get_fdata() > 0] = idx+1
        else:
            print(f"Missing: {file_in}")
    nib.save(nib.Nifti1Image(combined.astype(np.uint8), ref_img.affine), file_out)


if __name__ == "__main__":
    """
    Convert the downloaded TotalSegmentator dataset (after unzipping it) to nnUNet format and
    generate dataset.json and splits_final.json

    example usage:
    python convert_dataset_to_nnunet.py /my_downloads/TotalSegmentator_dataset /nnunet/raw/Dataset100_TotalSegmentator_part1 class_map_part_organs

    You must set nnUNet_raw and nnUNet_preprocessed environment variables before running this (see nnUNet documentation).
    """

    dataset_path = Path(sys.argv[1])  # directory containing all the subjects ie "/data/core-rad/data/totalsegmentator"
    nnunet_path = Path(sys.argv[2])  # directory of the new nnunet dataset ie "/data/core-rad/data/tucker/raw/008-lowres-remapped
    class_map_name = Path(sys.argv[3])
    # TotalSegmentator is made up of 5 models. Choose which one you want to produce. Choose from:
    #   class_map_part_organs
    #   class_map_part_vertebrae
    #   class_map_part_cardiac
    #   class_map_part_muscles
    #   class_map_part_ribs
    # Or for the 1.5mm lowres model 
    #   total
    class_map_name = "class_map_part_organs" 
    class_map_name = "class_map_part_vertebrae"
    class_map_name = "class_map_part_cardiac"
    class_map_name = "class_map_part_muscles"
    class_map_name = "class_map_part_ribs"
    class_map_name = "total"
    
    if class_map_name == "total":   
        class_map = class_map[class_map_name]
        class_map = {value: key for key, value in total_remap.items()} # There was a bug in the classmap in our TS version so we remapped the labels
    else:
        class_map = class_map_5_parts[class_map_name]
    
    (nnunet_path / "imagesTr").mkdir(parents=True, exist_ok=True)
    (nnunet_path / "labelsTr").mkdir(parents=True, exist_ok=True)
    (nnunet_path / "imagesTs").mkdir(parents=True, exist_ok=True)
    (nnunet_path / "labelsTs").mkdir(parents=True, exist_ok=True)

    meta = pd.read_csv(dataset_path / "meta.csv", sep=";")
    subjects_train = list(meta[meta["split"] == "train"]["image_id"].values)
    subjects_val = list(meta[meta["split"] == "val"]["image_id"].values)
    subjects_test = list(meta[meta["split"] == "test"]["image_id"].values)

    
    print("Copying train data...")
    subjects_tr = subjects_train + subjects_val
    def tr(idx):
        #for subject in tqdm(subjects_train + subjects_val):
        subject = subjects_tr[idx]
        subject_path = dataset_path / subject
        #shutil.copy(subject_path / "ct.nii.gz", nnunet_path / "imagesTr" / f"{subject}_0000.nii.gz")
        combine_labels(subject_path / "ct.nii.gz",
                       nnunet_path / "labelsTr" / f"{subject}.nii.gz",
                       [subject_path / "segmentations" / f"{roi}.nii.gz" for roi in class_map.values()])

    with multiprocessing.Pool(WORKER) as p:
        for i in tqdm(p.imap_unordered(tr, range(len(subjects_tr))), total=len(subjects_tr)):
            pass 
            
    print("Copying test data...")
    
    def t(idx):
        #for subject in tqdm(subjects_test):
        subject = subjects_test[idx]
        subject_path = dataset_path / subject
        #shutil.copy(subject_path / "ct.nii.gz", nnunet_path / "imagesTs" / f"{subject}_0000.nii.gz")
        combine_labels(subject_path / "ct.nii.gz",
                       nnunet_path / "labelsTs" / f"{subject}.nii.gz",
                       [subject_path / "segmentations" / f"{roi}.nii.gz" for roi in class_map.values()])

    with multiprocessing.Pool(WORKER) as p:
        for i in tqdm(p.imap_unordered(t, range(len(subjects_test))), total=len(subjects_test)):
            pass 

    generate_json_from_dir_v2(nnunet_path.name, subjects_train, subjects_val, class_map.values())
