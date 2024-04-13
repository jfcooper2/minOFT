import math

import torch
import torch.nn as nn
from torch.nn import init

from einops import rearrange

from src.models.layers.structured_linear import StructuredLinear
from src.models.layers.blockdiag_butterfly_multiply import blockdiag_butterfly_multiply

from src.utils.utils import get_logger
logger = get_logger()


class MonarchLinear(StructuredLinear):

    # def __init__(self, *args, nblocks=4, **kwargs):
    def __init__(self, *args, nblocks:int=4, adapt:bool=True, **kwargs):
        super().__init__(*args, **kwargs)
        in_blksz = int(math.ceil(self.in_features / nblocks))
        out_blksz = int(math.ceil(self.out_features / nblocks))
        self.in_features_extended = in_blksz * nblocks
        self.out_features_extended = out_blksz * nblocks
        self.nblocks = nblocks
        self.adapt = adapt

        if self.in_features_extended < self.out_features_extended:
            self.blkdiag1 = nn.Parameter(torch.empty(nblocks, in_blksz, in_blksz))
            self.blkdiag2 = nn.Parameter(torch.empty(nblocks, out_blksz, in_blksz))
        else:
            self.blkdiag1 = nn.Parameter(torch.empty(nblocks, out_blksz, in_blksz))
            self.blkdiag2 = nn.Parameter(torch.empty(nblocks, out_blksz, out_blksz))
        self.reset_parameters()
        logger.info(f'Linear class {self.__class__}: saving={self.saving}')

    def reset_parameters(self) -> None:
        if self.adapt:
            kaiming_init, zero_init = [self.blkdiag1] , [self.blkdiag2]
        else:
            kaiming_init, zero_init = [self.blkdiag1, self.blkdiag2], []
        # Mimic init.kaiming_uniform: https://github.com/pytorch/pytorch/blob/24087d07ca7ffa244575d259711dd7c99245a67a/torch/nn/init.py#L360
        # for blkdiag in [self.blkdiag1, self.blkdiag2]:
        for blkdiag in kaiming_init:
            fan_in = blkdiag.shape[-1]
            gain = init.calculate_gain(nonlinearity='leaky_relu', param=math.sqrt(5))
            std = gain / math.sqrt(fan_in)
            bound = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
            with torch.no_grad():
                blkdiag.uniform_(-bound, bound)
        for blkdiag in zero_init:
            with torch.no_grad():
                blkdiag.zero_()
        if not self.adapt:
            # bias in StructuredLinear is default 0s in __init__; reset_parameters_bias applies uniform
            self.reset_parameters_bias()

    @property
    def saving(self):
        return ((self.blkdiag1.numel() + self.blkdiag2.numel())
                / (self.in_features * self.out_features))

    def forward_matmul(self, x):
        output = blockdiag_butterfly_multiply(self.preprocess(x), self.blkdiag1, self.blkdiag2)
        return self.postprocess(output)
