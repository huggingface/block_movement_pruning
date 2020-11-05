# coding=utf-8
# Copyright 2020-present, the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Masked Linear module: A fully connected layer that computes an adaptive binary mask on the fly.
The mask (binary or not) is computed at each forward pass and multiplied against
the weight matrix to prune a portion of the weights.
The pruned weight matrix is then multiplied against the inputs (and if necessary, the bias is added).
"""

import math
from itertools import permutations

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import init

from .binarizer import MagnitudeBinarizer, ThresholdBinarizer, TopKBinarizer


class MaskedLinear(nn.Linear):
    """
    Fully Connected layer with on the fly adaptive mask.
    If needed, a score matrix is created to store the importance of each associated weight.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        mask_init: str = "constant",
        mask_scale: float = 0.0,
        pruning_method: str = "topK",
        mask_block_rows:int = 1,
        mask_block_cols:int = 1,
        ampere_pruning_method: str = "disabled",
        ampere_mask_init: str = "constant",
        ampere_mask_scale: float = 0.0,
    ):
        """
        Args:
            in_features (`int`)
                Size of each input sample
            out_features (`int`)
                Size of each output sample
            bias (`bool`)
                If set to ``False``, the layer will not learn an additive bias.
                Default: ``True``
            mask_init (`str`)
                The initialization method for the score matrix if a score matrix is needed.
                Choices: ["constant", "uniform", "kaiming"]
                Default: ``constant``
            mask_scale (`float`)
                The initialization parameter for the chosen initialization method `mask_init`.
                Default: ``0.``
            pruning_method (`str`)
                Method to compute the mask.
                Choices: ["topK", "threshold", "sigmoied_threshold", "magnitude", "l0"]
                Default: ``topK``
        """
        super(MaskedLinear, self).__init__(in_features=in_features, out_features=out_features, bias=bias)
        assert pruning_method in ["topK", "threshold", "sigmoied_threshold", "magnitude", "l0"]
        self.pruning_method = pruning_method
        self.mask_block_rows = mask_block_rows
        self.mask_block_cols = mask_block_cols
        assert ampere_pruning_method in ["disabled", "annealing"]
        self.ampere_pruning_method = ampere_pruning_method

        if self.pruning_method in ["topK", "threshold", "sigmoied_threshold", "l0"]:
            self.mask_scale = mask_scale
            self.mask_init = mask_init
            size = self.weight.size()
            assert(size[0] % self.mask_block_rows == 0)
            assert(size[1] % self.mask_block_cols == 0)
            mask_size = (size[0] // self.mask_block_rows, size[1] // self.mask_block_cols)
            self.mask_scores = nn.Parameter(torch.Tensor(size=mask_size))
            self.init_mask()

        if self.ampere_pruning_method == "annealing":
            self.ampere_mask_init = ampere_mask_init
            self.ampere_mask_scale = ampere_mask_scale
            self.initialize_ampere_weights()

    def ampere_pattern(self, device = None):
        if self.sparse_patterns is not None:
            if device is not None:
                if self.sparse_patterns.device != device:
                    self.sparse_patterns = self.sparse_patterns.to(device=device)
            return self.sparse_patterns
        patterns = torch.zeros(self.M)
        patterns[:self.N] = 1
        self.sparse_patterns = torch.Tensor(list(set(permutations(patterns.tolist()))))
        return self.sparse_patterns

    M = 4
    N = 2

    def initialize_ampere_weights(self):
        """"We must remember that weights are used in transposed form for forward pass,
        which we want to optimize the most.
        So we make sure we are creating an Ampere sparse pattern on the right dimension -> 0"""
        assert ((self.weight.shape[0] % self.M) == 0)
        self.sparse_patterns = None

        sparse_patterns_count = self.ampere_pattern(None).shape[0]
        # Creating the pattern in a transposed way to avoid a few ops later
        ampere_mask_size = (self.weight.shape[1], self.weight.shape[0] // self.M, sparse_patterns_count)
        self.ampere_weights = nn.Parameter(torch.Tensor(size=ampere_mask_size))

        if self.ampere_mask_init == "constant":
            init.constant_(self.ampere_weights, val=self.ampere_mask_scale)
        elif self.ampere_mask_init == "uniform":
            init.uniform_(self.ampere_weights, a=-self.ampere_mask_scale, b=self.ampere_mask_scale)
        elif self.ampere_mask_init == "kaiming":
            init.kaiming_uniform_(self.ampere_weights, a=math.sqrt(5))

    def ampere_mask(self, temperature:float, device):
        s = torch.nn.functional.softmax(self.ampere_weights * temperature, dim=-1)
        s = s.matmul(self.ampere_pattern(device))
        s = s.view(-1, s.shape[1] * s.shape[2])
        s = s.t()

        return s

    def init_mask(self):
        if self.mask_init == "constant":
            init.constant_(self.mask_scores, val=self.mask_scale)
        elif self.mask_init == "uniform":
            init.uniform_(self.mask_scores, a=-self.mask_scale, b=self.mask_scale)
        elif self.mask_init == "kaiming":
            init.kaiming_uniform_(self.mask_scores, a=math.sqrt(5))

    def expand_mask(self, mask):
        mask = torch.repeat_interleave(mask, self.mask_block_rows, dim=0)
        mask = torch.repeat_interleave(mask, self.mask_block_cols, dim=1)
        return mask

    def forward(self, input: torch.tensor, current_config: dict):
        # Get the mask
        threshold = current_config["threshold"]
        ampere_temperature = current_config["ampere_temperature"]
        if self.pruning_method == "topK":
            mask = TopKBinarizer.apply(self.mask_scores, threshold)
        elif self.pruning_method in ["threshold", "sigmoied_threshold"]:
            sig = "sigmoied" in self.pruning_method
            mask = ThresholdBinarizer.apply(self.mask_scores, threshold, sig)
        elif self.pruning_method == "magnitude":
            mask = MagnitudeBinarizer.apply(self.weight, threshold)
        elif self.pruning_method == "l0":
            l, r, b = -0.1, 1.1, 2 / 3
            if self.training:
                u = torch.zeros_like(self.mask_scores).uniform_().clamp(0.0001, 0.9999)
                s = torch.sigmoid((u.log() - (1 - u).log() + self.mask_scores) / b)
            else:
                s = torch.sigmoid(self.mask_scores)
            s_bar = s * (r - l) + l
            mask = s_bar.clamp(min=0.0, max=1.0)
        # Expand block mask to individual element mask
        if self.pruning_method != "magnitude":
            mask = self.expand_mask(mask)

        if self.ampere_pruning_method != "disabled":
            ampere_mask = self.ampere_mask(ampere_temperature, device=mask.device)
            mask = mask * ampere_mask

        # Mask weights with computed mask
        weight_thresholded = mask * self.weight
        # Compute output (linear layer) with masked weights
        return F.linear(input, weight_thresholded, self.bias)
