from batchgenerators.utilities.file_and_folder_operations import join, isfile
from nnunetv2.run.load_pretrained_weights import load_pretrained_weights
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from copy import deepcopy
from tuckercnn.tucker import DecompositionAgent
from typing import Optional
from dataclasses import dataclass
import torch
from torch.cuda.amp import GradScaler

TUCKER_ARGS = {
    'rank_mode': 'relative',
    'rank_factor': 1 / 3,
    'rank_min': None,
    'decompose': True,
    'verbose': True,
}


@dataclass
class MonkeyManager:
    tucker_args = TUCKER_ARGS
    apply_tucker = True
    inference_bs = 1
    ckpt_path = ''
    save_model = False
    load_model = False


from torch.optim.lr_scheduler import _LRScheduler


class PolyLRScheduler(_LRScheduler):
    def __init__(self, optimizer, initial_lr: float, max_steps: int, exponent: float = 0.9, current_step: int = None):
        self.optimizer = optimizer
        self.initial_lr = initial_lr
        self.max_steps = max_steps
        self.exponent = exponent
        self.ctr = 0
        super().__init__(optimizer, current_step if current_step is not None else -1, False)

    def step(self, current_step=None):
        if current_step is None or current_step == -1:
            current_step = self.ctr
            self.ctr += 1

        new_lr = self.initial_lr * (1 - current_step / self.max_steps) ** self.exponent
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr


def configure_optimizers(nntrainer):
    optimizer = torch.optim.SGD(nntrainer.network.parameters(), nntrainer.initial_lr,
                                weight_decay=nntrainer.weight_decay,
                                momentum=0.99, nesterov=True)
    lr_scheduler = PolyLRScheduler(optimizer, nntrainer.initial_lr, nntrainer.num_epochs)
    return optimizer, lr_scheduler


def maybe_load_checkpoint(nnunet_trainer: nnUNetTrainer, continue_training: bool, validation_only: bool,
                          pretrained_weights_file: str = None):
    if continue_training and pretrained_weights_file is not None:
        raise RuntimeError('Cannot both continue a training AND load pretrained weights. Pretrained weights can only '
                           'be used at the beginning of the training.')
    if continue_training:
        expected_checkpoint_file = join(nnunet_trainer.output_folder, 'checkpoint_final.pth')
        if not isfile(expected_checkpoint_file):
            expected_checkpoint_file = join(nnunet_trainer.output_folder, 'checkpoint_latest.pth')
        # special case where --c is used to run a previously aborted validation
        if not isfile(expected_checkpoint_file):
            expected_checkpoint_file = join(nnunet_trainer.output_folder, 'checkpoint_best.pth')
        if not isfile(expected_checkpoint_file):
            print(f"WARNING: Cannot continue training because there seems to be no checkpoint available to "
                  f"continue from. Starting a new training...")
            expected_checkpoint_file = None
    elif validation_only:
        expected_checkpoint_file = join(nnunet_trainer.output_folder, 'checkpoint_final.pth')
        if not isfile(expected_checkpoint_file):
            raise RuntimeError(f"Cannot run validation because the training is not finished yet!")
    else:
        if pretrained_weights_file is not None:
            if not nnunet_trainer.was_initialized:
                nnunet_trainer.initialize()

            if MonkeyManager.apply_tucker:
                load_pretrained_weights(nnunet_trainer.network, pretrained_weights_file, verbose=True)
                # nnunet_trainer.load_checkpoint(pretrained_weights_file)
                print("Start decomposition")
                print(nnunet_trainer.network)
                network = DecompositionAgent(
                    tucker_args=MonkeyManager.tucker_args,
                    ckpt_path=MonkeyManager.ckpt_path,
                    save_model=MonkeyManager.save_model,
                    load_model=MonkeyManager.load_model,
                )(deepcopy(nnunet_trainer.network))
                nnunet_trainer.network = network
                print("Start decomposition")
                print(nnunet_trainer.network)
                nnunet_trainer.grad_scaler = GradScaler()  # if self.device.type == 'cuda' else None
                nnunet_trainer.optimizer, nnunet_trainer.lr_scheduler = configure_optimizers(nnunet_trainer)
                nnunet_trainer.network.train()
            else:
                load_pretrained_weights(nnunet_trainer.network, pretrained_weights_file, verbose=True)
            # nnunet_trainer.load_checkpoint(pretrained_weights_file)
        expected_checkpoint_file = None

    if expected_checkpoint_file is not None:
        nnunet_trainer.load_checkpoint(expected_checkpoint_file)
