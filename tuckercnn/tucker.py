from typing import Optional

import numpy as np
import tensorly
import torch
from tensorly.decomposition import partial_tucker
from torch import nn
from torch.nn import Conv3d

import tuckercnn.VBMF as VBMF

from tuckercnn.utils import eprint

tensorly.set_backend('numpy')


class DecompositionAgent:
    def __init__(self, tucker_args: Optional[dict] = None, saved_model=False):
        self.tucker_args = tucker_args
        self.saved_model = saved_model
        # self.saved_ranks = saved_ranks

    def __call__(self, model: nn.Module) -> nn.Module:
        return self.apply(model)

    def apply(self, model: nn.Module) -> nn.Module:
        model = model.cpu()

        if self.saved_model:
            self.tucker_args['decompose'] = False
            replacer = LayerReplacer(tucker_args=self.tucker_args)
            LayerSurgeon(replacer).operate(model)
            model.load_state_dict(
                torch.load('/home/jakob/.totalsegmentator/tucker/model.pt')
            )
            model.eval()
        else:
            replacer = LayerReplacer(tucker_args=self.tucker_args)
            LayerSurgeon(replacer).operate(model)
            # torch.save(
            #     model.state_dict(), '/home/jakob/.totalsegmentator/tucker/model.pt'
            # )

        if torch.cuda.is_available():
            model = model.cuda()

        return model


class LayerReplacer:
    def __init__(self, tucker_args: Optional[dict] = None):
        self.targets = (Conv3d,)
        self.tucker_args = {} if tucker_args is None else tucker_args

    def should_replace(self, m: nn.Module) -> bool:
        return isinstance(m, self.targets)

    def get_replacement(self, m: nn.Module) -> nn.Module:
        if isinstance(m, Tucker):
            return m
        else:
            return Tucker(m, **self.tucker_args)


class Tucker(nn.Module):
    def __init__(
        self,
        m: nn.Module,
        rank_mode: str = 'relative',
        rank_factor: Optional[float] = 1 / 3,
        rank_min: Optional[int] = None,
        decompose: bool = True,
        verbose: bool = True,
    ):
        super().__init__()
        self.rank_mode = rank_mode
        self.rank_factor = rank_factor
        self.rank_min = rank_min
        self.decompose = decompose

        if decompose:
            self.ranks = self.get_ranks(m)
            self.seq = self.get_tucker_net(m)
            self.set_weights(m)
        else:
            self.ranks = self.get_ranks(m)
            self.seq = self.get_tucker_net(m)

        if verbose:
            eprint(
                f'Decomposed layer with Tucker. '
                f'[in, out] = [{m.in_channels:3d},  {m.out_channels:3d}] | '
                f'core [in, out] = [{self.seq[1].weight.shape[1]:3d},  '
                f'{self.seq[1].weight.shape[0]:3d}]'
            )

    def get_ranks(self, m: nn.Module) -> tuple[int, int]:
        if self.rank_mode == 'relative':
            assert (
                self.rank_factor is not None
            ), 'Argument rank_cr must be set when choosing rank_mode "relative".'

            new_in = int(np.ceil(m.in_channels * self.rank_factor))
            new_out = int(np.ceil(m.out_channels * self.rank_factor))
            ranks = [new_out, new_in]

        elif self.rank_mode == 'vmbf':
            ranks = estimate_ranks(m)

        elif self.rank_mode == 'fixed':
            assert (
                self.rank_min is not None
            ), 'Argument rank_min must be set when choosing rank_mode "fixed".'

            ranks = [self.rank_min] * 2

        else:
            raise ValueError(f'Rank mode {self.rank_mode} is unknown.')

        if self.rank_min is not None:
            ranks = np.maximum(self.rank_min, ranks)
            ranks[1] = min(m.in_channels, ranks[1])
            ranks[0] = min(m.out_channels, ranks[0])

        return ranks

    def get_tucker_net(self, m: nn.Module) -> nn.Sequential:
        use_bias = False if m.bias is None else True

        m_first = Conv3d(m.in_channels, int(self.ranks[1]), kernel_size=1, bias=False)
        m_core = Conv3d(
            in_channels=int(self.ranks[1]),
            out_channels=int(self.ranks[0]),
            kernel_size=m.kernel_size,
            bias=False,
            stride=m.stride,
            padding=m.padding,
            dilation=m.dilation,
            groups=m.groups,
        )
        m_last = Conv3d(
            int(self.ranks[0]), m.out_channels, kernel_size=1, bias=use_bias
        )

        return nn.Sequential(m_first, m_core, m_last)

    def set_weights(self, m: nn.Module) -> None:
        (core, (last, first)), _ = partial_tucker(
            tensor=m.weight.data.numpy(), modes=(0, 1), rank=self.ranks, init='svd'
        )

        core = torch.from_numpy(core)
        first = torch.from_numpy(first)
        last = torch.from_numpy(last)

        self.seq[0].weight.data = (
            torch.transpose(first, 1, 0).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        )
        self.seq[1].weight.data = core
        self.seq[2].weight.data = last.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        self.seq[2].bias.data = m.bias.data

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
