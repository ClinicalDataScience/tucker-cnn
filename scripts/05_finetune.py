import nnunetv2
from nnunetv2.run.run_training import run_training_entry
from tuckercnn.monkey_patch_train import maybe_load_checkpoint
from unittest.mock import patch
import os

if __name__ == '__main__':
    os.environ["nnUNet_preprocessed"] = "/data/core-rad/data/tucker/preprocessed"
    os.environ["nnUNet_raw"] = "/data/core-rad/data/tucker/raw"
    os.environ["nnUNet_results"] = "/data/core-rad/data/tucker/results"
    # export nnUNet_preprocessed="/data/core-rad/data/tucker/preprocessed"
    # export nnUNet_raw="/data/core-rad/data/tucker/raw"
    # export nnUNet_results="/data/core-rad/data/tucker/results"
    # export nnUNet_n_proc_DA=28
    # setup nnUNetv2_plan_and_preprocess -d 001 -pl ExperimentPlanner -c 3d_fullres -np 2
    # copy total seg plan to file and change ds, then
    # nnUNetv2_preprocess -d 001 -plans_name /data/core-rad/data/tucker/preprocessed/Dataset001-torgan/nnUNetPlans -c 3d_fullres -np 28

    nnunetv2.run.run_training.maybe_load_checkpoint = maybe_load_checkpoint

    run_args = ['', '300', '3d_fullres', '0', '-tr', 'nnUNetTrainerNoMirroring']
    run_args = ['', '001', '3d_fullres', '0', '-tr', 'nnUNetTrainerNoMirroring', '-pretrained_weights',
                '/data/core-rad/data/tucker/results/Dataset291_TotalSegmentator_part1_organs_1559subj/nnUNetTrainerNoMirroring__nnUNetPlans__3d_fullres/fold_0/checkpoint_final.pth']
    run_args = ['', '006', '3d_fullres', '0', '-tr', 'nnUNetTrainerNoMirroring', '-pretrained_weights',
                '/data/core-rad/data/tucker/results/Dataset297_TotalSegmentator_total_3mm_1559subj/nnUNetTrainer_4000epochs_NoMirroring__nnUNetPlans__3d_fullres/fold_0/checkpoint_final.pth']

    with patch("sys.argv", run_args):
        run_training_entry()
    # or run with sysargs like nnunet_train
    # nnUNetv2_train <your_dataset_id> 3d_fullres 0 -tr nnUNetTrainerNoMirroring
    # aka run_training_entry(sys.arv)
    # parser = argparse.ArgumentParser()
    # subparsers = parser.add_subparsers()
    # nnrun = subparsers.add_parser('run_training_entry', help='runnn')
    # nnrun.set_defaults(func=run_training_entry)
    # run_args = '300', '3d_fullres', '0', '-tr', 'nnUNetTrainerNoMirroring'
    # sys.argv.append(run_args)
    # options = parser.parse_args()
    #
    # options.func()