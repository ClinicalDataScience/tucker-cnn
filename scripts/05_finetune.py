import nnunetv2
from nnunetv2.run.run_training import run_training_entry
from tuckercnn.monkey_patch_train import maybe_load_checkpoint
from unittest.mock import patch

if __name__ == '__main__':
    #nnunetv2.run.run_training.maybe_load_checkpoint = maybe_load_checkpoint
    #either
    run_args = ['', '300', '3d_fullres', '0', '-tr', 'nnUNetTrainerNoMirroring']
    with patch("sys.argv", run_args):
        run_training_entry()
    # or run with sysargs like nnunet_train
    #nnUNetv2_train <your_dataset_id> 3d_fullres 0 -tr nnUNetTrainerNoMirroring
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