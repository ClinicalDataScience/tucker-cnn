import sys

import SimpleITK as sitk
from dynamic_network_architectures.building_blocks.simple_conv_blocks import (
    ConvDropoutNormReLU,
)
from dynamic_network_architectures.building_blocks.unet_decoder import UNetDecoder
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
