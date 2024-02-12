from batchgenerators.utilities.file_and_folder_operations import join, isfile
from nnunetv2.run.load_pretrained_weights import load_pretrained_weights
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from copy import deepcopy
from tuckercnn.tucker import DecompositionAgent
from typing import Optional
from dataclasses import dataclass


@dataclass
class MonkeyManager:
    tucker_args: Optional[dict] = None
    apply_tucker = True
    inference_bs = 1
    ckpt_path = ''
    save_model = False
    load_model = False


def maybe_load_checkpoint(
    nnunet_trainer: nnUNetTrainer,
    continue_training: bool,
    validation_only: bool,
    pretrained_weights_file: str = None,
):
    if continue_training and pretrained_weights_file is not None:
        raise RuntimeError(
            'Cannot both continue a training AND load pretrained weights. Pretrained weights can only '
            'be used at the beginning of the training.'
        )
    if continue_training:
        expected_checkpoint_file = join(
            nnunet_trainer.output_folder, 'checkpoint_final.pth'
        )
        if not isfile(expected_checkpoint_file):
            expected_checkpoint_file = join(
                nnunet_trainer.output_folder, 'checkpoint_latest.pth'
            )
        # special case where --c is used to run a previously aborted validation
        if not isfile(expected_checkpoint_file):
            expected_checkpoint_file = join(
                nnunet_trainer.output_folder, 'checkpoint_best.pth'
            )
        if not isfile(expected_checkpoint_file):
            print(
                f"WARNING: Cannot continue training because there seems to be no checkpoint available to "
                f"continue from. Starting a new training..."
            )
            expected_checkpoint_file = None
    elif validation_only:
        expected_checkpoint_file = join(
            nnunet_trainer.output_folder, 'checkpoint_final.pth'
        )
        if not isfile(expected_checkpoint_file):
            raise RuntimeError(
                f"Cannot run validation because the training is not finished yet!"
            )
    else:
        if pretrained_weights_file is not None:
            if not nnunet_trainer.was_initialized:
                nnunet_trainer.initialize()

            if MonkeyManager.apply_tucker:
                network = DecompositionAgent(
                    tucker_args=MonkeyManager.tucker_args,
                    ckpt_path=MonkeyManager.ckpt_path,
                    save_model=MonkeyManager.save_model,
                    load_model=MonkeyManager.load_model,
                )(deepcopy(nnunet_trainer.network))
                nnunet_trainer.network = network
            else:
                load_pretrained_weights(
                    nnunet_trainer.network, pretrained_weights_file, verbose=True
                )
        expected_checkpoint_file = None

    if expected_checkpoint_file is not None:
        nnunet_trainer.load_checkpoint(expected_checkpoint_file)
