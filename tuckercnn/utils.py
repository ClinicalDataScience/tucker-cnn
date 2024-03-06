import sys

import nibabel as nib
import numpy as np
from dynamic_network_architectures.building_blocks.simple_conv_blocks import (
    ConvDropoutNormReLU,
)
from dynamic_network_architectures.building_blocks.unet_decoder import UNetDecoder
from torch import nn
from surface_distance import (
    compute_surface_distances,
    compute_surface_dice_at_tolerance,
)
import yaml


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def read_nii(path: str) -> np.ndarray:
    return nib.load(path).get_fdata()


def read_yml(path: str) -> dict:
    with open(path, 'r') as file:
        return yaml.load(file, Loader=yaml.SafeLoader)


def save_yml(data, filename):
    with open(filename, 'w') as file:
        yaml.dump(data, file, default_flow_style=False)


def get_dice_score(gt: np.ndarray, pred: np.ndarray) -> float:
    overlap = (gt * pred).sum()
    sum = gt.sum() + pred.sum()
    dice_score = 2 * overlap / (sum + 1e-6)
    return float(dice_score)


def get_surface_distance(
    gt: np.ndarray, pred: np.ndarray, spacing: float = 1.5, tolerance: float = 3
) -> float:
    sd = compute_surface_distances(gt.astype(bool), pred.astype(bool), [spacing] * 3)
    nsd = compute_surface_dice_at_tolerance(sd, tolerance)
    return float(nsd)


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
