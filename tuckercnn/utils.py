import sys

from dynamic_network_architectures.building_blocks.simple_conv_blocks import (
    ConvDropoutNormReLU,
)
from dynamic_network_architectures.building_blocks.unet_decoder import UNetDecoder
import numpy as np
import SimpleITK as sitk
import torch
from torch import nn


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


def get_batch_iterable(iterable, n):
    # from https://stackoverflow.com/questions/5389507/iterating-over-every-two-elements-in-a-list
    "s -> (s0,s1,s2,...sn-1), (sn,sn+1,sn+2,...s2n-1), (s2n,s2n+1,s2n+2,...s3n-1), ..."
    return zip(*[iter(iterable)] * n)


class Timer:
    execution_times = []
    verbose = True
    warm_up_rounds = 1

    def __init__(self):
        self.start = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)

    def __enter__(self):
        self.start.record()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.end.record()
        torch.cuda.synchronize()

        ms = self.start.elapsed_time(self.end)

        if Timer.verbose:
            eprint(f'Elapsed time: {ms:7.2f}ms')

        Timer.execution_times.append(ms)

    @classmethod
    def report(cls) -> None:
        if len(cls.execution_times) == 0:
            print('Nothing to report. No times were recorded.')
            return

        mean = float(np.mean(cls.execution_times[cls.warm_up_rounds :]))
        std = float(np.std(cls.execution_times[cls.warm_up_rounds :]))

        print(
            f'Execution took {mean:.2f}±{std:.2f}ms over '
            f'{len(cls.execution_times) - cls.warm_up_rounds} function calls.'
        )

        if cls.warm_up_rounds == 1:
            print('1 function call was ignored in this report due to warm up.')
        elif cls.warm_up_rounds > 1:
            print(
                f'{cls.warm_up_rounds} function calls were ignored in '
                f'this report due to warm up.'
            )

    @classmethod
    def reset(cls) -> None:
        cls.execution_times.clear()


def streamline_nnunet_architecture(network: nn.Module) -> nn.Module:
    """Removes redundant attributes from the network."""
    for m in network.modules():
        if isinstance(m, ConvDropoutNormReLU):
            if hasattr(m, 'conv'):
                delattr(m, 'conv')
            if hasattr(m, 'dropout'):
                delattr(m, 'dropout')
            if hasattr(m, 'norm'):
                delattr(m, 'norm')
            if hasattr(m, 'nonlin'):
                delattr(m, 'nonlin')

        if isinstance(m, UNetDecoder):
            if not m.deep_supervision:
                m.seg_layers = nn.ModuleList([m.seg_layers[-1]])

    return network
