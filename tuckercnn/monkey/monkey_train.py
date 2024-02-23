import numpy as np
import types
import torch

from torch.cuda.amp import GradScaler
from torch.optim.lr_scheduler import _LRScheduler
from torch._dynamo import OptimizedModule
from torch.nn.parallel import DistributedDataParallel as DDP

from copy import deepcopy
from tqdm import tqdm

from batchgenerators.utilities.file_and_folder_operations import join, isfile
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

from tuckercnn.tucker import DecompositionAgent
from tuckercnn.monkey.config import MonkeyConfig


LR_FACTOR = 1


def run_training(self):
    self.on_train_start()

    for epoch in range(self.current_epoch, self.num_epochs):
        self.on_epoch_start()

        # get initial score
        if epoch == 0:
            with torch.no_grad():
                self.on_validation_epoch_start()
                val_outputs = []
                for batch_id in range(self.num_val_iterations_per_epoch):
                    val_outputs.append(self.validation_step(next(self.dataloader_val)))
                self.on_validation_epoch_end(val_outputs)
                self.print_to_log_file('val_loss',
                                       np.round(self.logger.my_fantastic_logging['val_losses'][-1], decimals=4))
                self.print_to_log_file('Pseudo dice', [np.round(i, decimals=4) for i in
                                                       self.logger.my_fantastic_logging['dice_per_class_or_region'][
                                                           -1]])
                self.print_to_log_file('Mean Pseudo dice', np.nanmean([i for i in
                                                                       self.logger.my_fantastic_logging[
                                                                           'dice_per_class_or_region'][-1]]))
                self._best_ema = self.logger.my_fantastic_logging['ema_fg_dice'][-1]
                self.print_to_log_file(f"EMA pseudo Dice: {np.round(self._best_ema, decimals=4)}")

        self.on_train_epoch_start()
        train_outputs = []
        for batch_id in tqdm(range(self.num_iterations_per_epoch)):
            train_outputs.append(self.train_step(next(self.dataloader_train)))
        self.on_train_epoch_end(train_outputs)

        with torch.no_grad():
            self.on_validation_epoch_start()
            val_outputs = []
            for batch_id in range(self.num_val_iterations_per_epoch):
                val_outputs.append(self.validation_step(next(self.dataloader_val)))
            self.on_validation_epoch_end(val_outputs)

        self.on_epoch_end()

    self.on_train_end()


def load_pretrained_weights(network, fname, verbose=False):
    """
    Transfers all weights between matching keys in state_dicts. matching is done by name and we only transfer if the
    shape is also the same. Segmentation layers (the 1x1(x1) layers that produce the segmentation maps)
    identified by keys ending with '.seg_layers') are not transferred!

    If the pretrained weights were obtained with a training outside nnU-Net and DDP or torch.optimize was used,
    you need to change the keys of the pretrained state_dict. DDP adds a 'module.' prefix and torch.optim adds
    '_orig_mod'. You DO NOT need to worry about this if pretraining was done with nnU-Net as
    nnUNetTrainer.save_checkpoint takes care of that!

    """
    saved_model = torch.load(fname)
    pretrained_dict = saved_model['network_weights']

    skip_strings_in_pretrained = [  # TODO
        # '.seg_layers.',
    ]

    if isinstance(network, DDP):
        mod = network.module
    else:
        mod = network
    if isinstance(mod, OptimizedModule):
        mod = mod._orig_mod

    model_dict = mod.state_dict()
    # verify that all but the segmentation layers have the same shape
    for key, _ in model_dict.items():
        if all([i not in key for i in skip_strings_in_pretrained]):
            assert key in pretrained_dict, (
                f"Key {key} is missing in the pretrained model weights. The pretrained weights do not seem to be "
                f"compatible with your network."
            )
            assert model_dict[key].shape == pretrained_dict[key].shape, (
                f"The shape of the parameters of key {key} is not the same. Pretrained model: "
                f"{pretrained_dict[key].shape}; your network: {model_dict[key]}. The pretrained model "
                f"does not seem to be compatible with your network."
            )

    # fun fact: in principle this allows loading from parameters that do not cover the entire network. For example pretrained
    # encoders. Not supported by this function though (see assertions above)

    # commenting out this abomination of a dict comprehension for preservation in the archives of 'what not to do'
    # pretrained_dict = {'module.' + k if is_ddp else k: v
    #                    for k, v in pretrained_dict.items()
    #                    if (('module.' + k if is_ddp else k) in model_dict) and
    #                    all([i not in k for i in skip_strings_in_pretrained])}

    pretrained_dict = {
        k: v
        for k, v in pretrained_dict.items()
        if k in model_dict.keys()
        and all([i not in k for i in skip_strings_in_pretrained])
    }

    model_dict.update(pretrained_dict)

    print(
        "################### Loading pretrained weights from file ",
        fname,
        '###################',
    )
    if verbose:
        print(
            "Below is the list of overlapping blocks in pretrained model and nnUNet architecture:"
        )
        for key, value in pretrained_dict.items():
            print(key, 'shape', value.shape)
        print("################### Done ###################")
    mod.load_state_dict(model_dict)


class PolyLRScheduler(_LRScheduler):
    def __init__(
        self,
        optimizer,
        initial_lr: float,
        max_steps: int,
        exponent: float = 0.9,
        current_step: int = None,
    ):
        self.optimizer = optimizer
        self.initial_lr = initial_lr
        self.max_steps = max_steps
        self.exponent = exponent
        self.ctr = 0
        super().__init__(
            optimizer, current_step if current_step is not None else -1, False
        )

    def step(self, current_step=None):
        if current_step is None or current_step == -1:
            current_step = self.ctr
            self.ctr += 1

        new_lr = self.initial_lr * (1 - current_step / self.max_steps) ** self.exponent
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr


class DummyScheduler(_LRScheduler):  # TODO
    def __init__(
        self,
        optimizer,
        initial_lr: float,
        max_steps: int,
        exponent: float = 0.9,
        current_step: int = None,
    ):
        self.optimizer = optimizer
        self.initial_lr = initial_lr
        self.max_steps = max_steps
        self.exponent = exponent
        self.ctr = 0
        super().__init__(
            optimizer, current_step if current_step is not None else -1, False
        )

    def step(self, current_step=None):
        pass


def configure_optimizers(nntrainer):  # TODO
    nntrainer.initial_lr = nntrainer.initial_lr * LR_FACTOR
    optimizer = torch.optim.SGD(
        nntrainer.network.parameters(),
        nntrainer.initial_lr,
        weight_decay=nntrainer.weight_decay,
        momentum=0.99,
        nesterov=True,
    )
    lr_scheduler = DummyScheduler(optimizer, nntrainer.initial_lr, nntrainer.num_epochs)
    return optimizer, lr_scheduler


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

            if MonkeyConfig.apply_tucker:
                load_pretrained_weights(
                    nnunet_trainer.network, pretrained_weights_file, verbose=True
                )
                # nnunet_trainer.load_checkpoint(pretrained_weights_file)
                print("Start decomposition")
                # print(nnunet_trainer.network)
                network = DecompositionAgent(
                    tucker_args=MonkeyConfig.tucker_args,
                    ckpt_path=MonkeyConfig.ckpt_path,
                    save_model=MonkeyConfig.save_model,
                    load_model=MonkeyConfig.load_model,
                )(deepcopy(nnunet_trainer.network))
                nnunet_trainer.network = network
                # print(nnunet_trainer.network)
                nnunet_trainer.grad_scaler = (
                    GradScaler()
                )  # if self.device.type == 'cuda' else None #TODO
                nnunet_trainer.optimizer, nnunet_trainer.lr_scheduler = (
                    configure_optimizers(nnunet_trainer)
                )  # TODO
                nnunet_trainer.network.train()
            else:
                load_pretrained_weights(
                    nnunet_trainer.network, pretrained_weights_file, verbose=True
                )
                nnunet_trainer.grad_scaler = (
                    GradScaler()
                )  # if self.device.type == 'cuda' else None #TODO
                nnunet_trainer.optimizer, nnunet_trainer.lr_scheduler = (
                    configure_optimizers(nnunet_trainer)
                )  # TODO
            # nnunet_trainer.load_checkpoint(pretrained_weights_file)
        expected_checkpoint_file = None

    # TODO
    # patch train fn to print initial val dice pseudos
    nnunet_trainer.run_training = types.MethodType(run_training, nnunet_trainer)

    if expected_checkpoint_file is not None:
        nnunet_trainer.load_checkpoint(expected_checkpoint_file)
