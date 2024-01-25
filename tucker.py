import sys

import sys
from typing import Optional

import numpy as np
import tensorly
import torch
from tensorly.decomposition import partial_tucker
from torch import nn
from torch.nn import Conv3d

import VBMF

from utils import eprint

tensorly.set_backend('numpy')


class Decomposer:
    def __init__(
        self,
        rank_mode: str = 'heuristic',
        rank: Optional=None,
        tmp_dir=".decomposed_models/",
        debug=False,
    ):
        self.rank_mode = rank_mode
        self.rank = rank
        self.tmp_dir = tmp_dir
        self.debug = debug

    def __call__(self, model):
        return self.decomposer(model)

    def decomposer(self, model, name=None):
        model = model.cpu()
        replacer = LayerReplacer(self.rank_mode, self.rank)
        LayerSurgeon(replacer).operate(model)
        model = model.cuda()
        eprint(
            "Patch test ---------------------------------------------------------------------------------"
        )
        return model


class LayerReplacer:
    def __init__(self, rank_mode, rank):
        self.targets = (Conv3d, )
        self.rank = rank
        self.rank_mode = rank_mode

    def should_replace(self, m: nn.Module) -> bool:
        return isinstance(m, self.targets)

    def get_replacement(self, m: nn.Module) -> nn.Module:
        if isinstance(m, Tucker):
            return m
        else:
            return Tucker(m, self.rank_mode, self.rank)


class Tucker(nn.Module):
    def __init__(
            self, m: nn.Module,
            rank_mode="heuristic",
            rank_min: Optional[int] = None
    ):
        super().__init__()
        self.rank_mode = rank_mode
        self.rank_min = rank_min

        self.ranks = self.get_ranks(m)

        use_bias = False if m.bias is None else True

        (middle, (last, first)), rec_errors = partial_tucker(
            m.weight.data.numpy(), modes=(0, 1), rank=self.ranks, init='svd'
        )
        # eprint(first, last)

        middle = torch.from_numpy(middle)
        first = torch.from_numpy(first)
        last = torch.from_numpy(last)

        m_first = Conv3d(m.in_channels, self.ranks[1], kernel_size=1, bias=False)
        m_middle = m.__class__(
            in_channels=self.ranks[1],
            out_channels=self.ranks[0],
            kernel_size=m.kernel_size,
            bias=False,
            stride=m.stride,
            padding=m.padding,
            dilation=m.dilation,
            groups=m.groups,
        )
        m_last = Conv3d(self.ranks[0], m.out_channels, kernel_size=1, bias=use_bias)

        m_first.weight.data = (
            torch.transpose(first, 1, 0).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        )
        m_middle.weight.data = middle
        m_last.weight.data = last.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        m_last.bias.data = m_last.bias.data

        self.seq = nn.Sequential(m_first, m_middle, m_last)

        eprint('Decomposed layer. Rank: ', self.ranks, m)

    def get_ranks(self, m: nn.Module) -> tuple[int, int]:
        if self.rank_mode == 'heuristic':
            r3 = int(np.ceil(m.in_channels / 3))  # s/3
            r4 = int(np.ceil(m.out_channels / 3))  # t/3
            ranks = [r4, r3]
            ranks = np.maximum(16, ranks)
        elif self.rank_mode == 'vmbf':
            ranks = estimate_ranks(m)

        elif self.rank_mode == 'vmbf_min':
            ranks = estimate_ranks(m)
            if self.rank_min is None:
                self.rank_min = 64
            ranks = np.maximum(self.rank_min, ranks)

        elif self.rank_mode == 'fixed':
            ranks = [self.rank_min] * 2

        else:
            print('rank mode not defined')
            sys.exit()

        return ranks

    def forward(self, x):
        return self.seq(x)


def estimate_ranks(layer):
    """Unfold the 2 modes of the Tensor the decomposition will
    be performed on, and estimates the ranks of the matrices using VBMF
    """

    weights = layer.weight.data.numpy()
    unfold_0 = tensorly.base.unfold(weights, 0)
    unfold_1 = tensorly.base.unfold(weights, 1)
    _, diag_0, _, _ = VBMF.EVBMF(unfold_0)
    _, diag_1, _, _ = VBMF.EVBMF(unfold_1)
    ranks = [diag_0.shape[0], diag_1.shape[1]]
    return ranks


class LayerSurgeon:
    def __init__(self, replacer: LayerReplacer):
        self.replacer = replacer

    def operate(self, model: nn.Module) -> nn.Module:
        self._replace_layers(model)
        return model

    def _replace_layers(self, module: nn.Module) -> None:
        """
        Replace all target layers recursively.

        Recursively iterate over the module's layers. If an attribute is a convertible
        instance, it will be replaced inplace with the matching layer.
        The procedure is inspired by :
        https://discuss.pytorch.org/t/how-to-replace-a-layer-with-own-custom-variant/43586/7
        """
        new_layer: nn.Module
        # print(module.__class__)

        if isinstance(module, Tucker):
            return

        # Accessing layers over attributes does not work if it is an instance of
        # nn.Sequential. In this case we can access the respective layers over
        # indexing the Sequential`s children.
        if isinstance(module, (nn.Sequential, nn.ModuleList)):
            for i, child_module in enumerate(module.children()):
                # If a child module is a convertible layer, it is replaced.
                if self.replacer.should_replace(child_module):
                    module[i] = self.replacer.get_replacement(child_module)
        else:
            # Iterate over all attributes in the module
            for attr_str in dir(module):
                target_attr = getattr(module, attr_str)

                # If an attribute is a convertible instance, a new object is created and
                # is set as the corresponding attribute.
                if self.replacer.should_replace(target_attr):
                    new_layer = self.replacer.get_replacement(target_attr)
                    setattr(module, attr_str, new_layer)

        # Recursively iterate over children modules.
        for child_module in module.children():
            self._replace_layers(child_module)
