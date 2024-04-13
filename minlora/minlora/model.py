"""
References:
1) the official LoRA implementation released by Microsoft:
https://github.com/microsoft/LoRA/blob/main/loralib/layers.py
"""

import math
from functools import partial

import torch
import torch.nn.utils.parametrize as parametrize
from torch import nn

from .blockdiag_butterfly_multiply import blockdiag_butterfly_multiply


class LoRAParametrization(nn.Module):
    def __init__(self, fan_in, fan_out, fan_in_fan_out=False, rank=4, num_blocks=(4,4), lora_dropout_p=0.0, lora_alpha=1):
        super().__init__()
        # if weight is stored as (fan_out, fan_in), the memory layout of A & B follows (W + BA)x
        # otherwise, it's x(W + AB). This allows us to tie the weights between linear layers and embeddings
#         self.swap = (lambda x: (x[1], x[0])) if fan_in_fan_out else (lambda x: x)
        
        self.fan_in, self.fan_out, self.rank = fan_in, fan_out, rank
        
        # num_blocks_in   : k
        # blk_height_in   : q
        # blk_width_in    : p
        # num_blocks_out  : l
        # blk_height_out  : s
        # blk_width_out   : r
        # btfly_in        : k*p (close to fan_in)
        # btfly_out       : l*s (close to fan_out)
        # rank            : k*q=l*r
        if type(num_blocks) == int:
            self.num_blocks_in, self.num_blocks_out = num_blocks, num_blocks
        else:
            self.num_blocks_in, self.num_blocks_out = num_blocks # Tuple
        
        assert int(self.num_blocks_in) % int(self.rank) == 0 and int(self.num_blocks_out) % int(self.rank) == 0
        
        self.blk_height_in = int(self.rank) // int(self.num_blocks_in)
        self.blk_width_in = int(math.ceil(self.fan_in / self.num_blocks_in))
        self.blk_height_out = int(math.ceil(self.fan_in / self.num_blocks_in))
        self.blk_width_out = int(self.rank) // int(self.num_blocks_out)
        self.btfly_in, self.btfly_out = self.num_blocks_in * self.blk_width_in, self.num_blocks_out * self.blk_height_out
        
        self.lora_A = nn.Parameter(torch.zeros((self.num_blocks_in, self.blk_height_in, self.blk_width_in)))
        self.lora_B = nn.Parameter(torch.zeros((self.num_blocks_out, self.blk_height_out, self.blk_width_out)))
#         self.lora_A = nn.Parameter(torch.zeros(self.swap((rank, fan_in))))
#         self.lora_B = nn.Parameter(torch.zeros(self.swap((fan_out, rank))))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / rank
        self.lora_dropout = nn.Dropout(p=lora_dropout_p) if lora_dropout_p > 0 else lambda x: x
        self.dropout_fn = self._dropout if lora_dropout_p > 0 else lambda x: x
        self.register_buffer("lora_dropout_mask", torch.ones((1, fan_in), dtype=self.lora_A.dtype))
#         self.register_buffer("lora_dropout_mask", torch.ones(self.swap((1, fan_in)), dtype=self.lora_A.dtype))
        self.forward_fn = self.lora_forward

    def _dropout(self, A):
        # to mimic the original implementation: A @ dropout(x), we do (A * dropout(ones)) @ x
        return A * self.lora_dropout(self.lora_dropout_mask)

    def lora_forward(self, X):
        print(X.shape)
        return X + (torch.matmul(self.lora_B, self.dropout_fn(self.lora_A)).T).view(X.shape) * self.scaling
#         return X + torch.matmul(*self.swap((self.lora_B, self.dropout_fn(self.lora_A)))).view(X.shape) * self.scaling

    def forward(self, X):
        return self.forward_fn(X)

    def disable_lora(self):
        self.forward_fn = lambda x: x

    def enable_lora(self):
        self.forward_fn = self.lora_forward

    @classmethod
    def from_linear(cls, layer, rank=4, lora_dropout_p=0.0, lora_alpha=1):
        fan_out, fan_in = layer.weight.shape
        return cls(
            fan_in, fan_out, fan_in_fan_out=False, rank=rank, lora_dropout_p=lora_dropout_p, lora_alpha=lora_alpha
        )

    @classmethod
    def from_conv2d(cls, layer, rank=4, lora_dropout_p=0.0, lora_alpha=1):
        fan_out, fan_in = layer.weight.view(layer.weight.shape[0], -1).shape
        return cls(
            fan_in, fan_out, fan_in_fan_out=False, rank=rank, lora_dropout_p=lora_dropout_p, lora_alpha=lora_alpha
        )

    @classmethod
    def from_embedding(cls, layer, rank=4, lora_dropout_p=0.0, lora_alpha=1):
        fan_in, fan_out = layer.weight.shape
        return cls(
            fan_in, fan_out, fan_in_fan_out=True, rank=rank, lora_dropout_p=lora_dropout_p, lora_alpha=lora_alpha
        )


default_lora_config = {  # specify which layers to add lora to, by default only add to linear layers
    nn.Linear: {
        "weight": partial(LoRAParametrization.from_linear, rank=4),
    },
}


def apply_lora(layer, register=True, merge=False, lora_config=default_lora_config):
    """add lora parametrization to a layer, designed to be used with model.apply"""
    if register:
        if type(layer) in lora_config:
            for attr_name, parametrization in lora_config[type(layer)].items():
                parametrize.register_parametrization(layer, attr_name, parametrization(layer))
    else:  # this will remove all parametrizations, use with caution
        if hasattr(layer, "parametrizations"):
            for attr_name in layer.parametrizations.keys():
                parametrize.remove_parametrizations(layer, attr_name, leave_parametrized=merge)


def add_lora(model, lora_config=default_lora_config):
    """add lora parametrization to all layers in a model. Calling it twice will add lora twice"""
    model.apply(partial(apply_lora, lora_config=lora_config))


def add_lora_by_name(model, target_module_names, lora_config=default_lora_config):
    """Add LoRA parameterization to specific layers in a model by names"""
    for name, layer in model.named_modules():
        if any([m in name for m in target_module_names]):
            add_lora(layer, lora_config=lora_config)


def merge_lora(model):
    """merge lora parametrization to all layers in a model. This will remove all parametrization"""
    model.apply(partial(apply_lora, register=False, merge=True))


def remove_lora(model):
    """remove lora parametrization to all layers in a model. This will remove all parametrization"""
    model.apply(partial(apply_lora, register=False, merge=False))
