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

import itertools
import math
import random
from itertools import permutations

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import init

from .binarizer import MagnitudeBinarizer, ThresholdBinarizer, TopKBinarizer

sparse_patterns = None

AMPERE_M = 4
AMPERE_N = 2


class StaticIndexDim1(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, index, reverse_index):
        ctx.save_for_backward(reverse_index)
        return input[:, index]

    @staticmethod
    def backward(ctx, grad_output):
        reverse_index, = ctx.saved_tensors
        return grad_output[:, reverse_index], None, None


class Index0(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, index, reverse_index):
        ctx.save_for_backward(reverse_index)
        return input[index, :]

    @staticmethod
    def backward(ctx, grad_output):
        reverse_index, = ctx.saved_tensors
        return grad_output[reverse_index, :], None, None


def ampere_pattern(device=None):
    global sparse_patterns, AMPERE_N, AMPERE_M
    if sparse_patterns is not None:
        if device is not None:
            if sparse_patterns.device != device:
                sparse_patterns = sparse_patterns.to(device=device)
        return sparse_patterns
    patterns = torch.zeros(AMPERE_M)
    patterns[:AMPERE_N] = 1
    sparse_patterns = torch.Tensor(list(set(permutations(patterns.tolist()))))
    return sparse_patterns

class DimensionShuffler(nn.Module):
    def __init__(self, in_features, out_features, in_features_group = 4, out_features_group = 4):
        super().__init__()
        self.in_features = in_features
        self.in_features_group = in_features_group
        self.out_features = out_features
        self.out_features_group = out_features_group

        in_mapping = self.dimension_mapping(in_features)
        out_mapping = self.dimension_mapping(out_features)
        out_mapping_reverse = out_mapping.sort()[1]

        self.register_buffer("in_mapping", in_mapping)
        self.register_buffer("out_mapping", out_mapping)
        self.register_buffer("out_mapping_reverse", out_mapping_reverse)

        #in_permutations = self.all_permutations(in_features_group)[2]
        #self.register_buffer("in_permutations", in_permutations)

        #out_permutations = self.all_permutations(out_features_group)[2]
        #self.register_buffer("out_permutations", out_permutations)

        in_permutation_scores = torch.randn(in_features // in_features_group, in_features_group - 1)
        out_permutation_scores = torch.randn(out_features // out_features_group, out_features_group - 1)

#        self.register_buffer("in_permutation_scores", in_permutation_scores)
#        self.register_buffer("out_permutation_scores", out_permutation_scores)
        self.in_permutation_scores = nn.Parameter(in_permutation_scores)
        self.out_permutation_scores = nn.Parameter(out_permutation_scores)

    @staticmethod
    def rotate_matrices(angles):
        assert(angles.shape[-1] == 1)
        c = angles.cos()
        s = angles.sin()

        rot0 = torch.cat([c, -s], dim=1)
        rot1 = torch.cat([s, c], dim=1)
        rot = torch.stack([rot0, rot1], dim=1)
        return rot, rot.transpose(1, 2)

    def forward(self, input, weight, mask, temperature):
        in_permutations, in_permutations_inverse = self.rotate_matrices(self.in_permutation_scores)
        out_permutations, out_permutations_inverse = self.rotate_matrices(self.out_permutation_scores)
        #in_permutations = self.permutation_mix(self.in_permutation_scores, self.in_permutations, temperature, self.training)
        #out_permutations = self.permutation_mix(self.out_permutation_scores, self.out_permutations, temperature, self.training)

        return self.permutated_linear(input,
                                      self.in_mapping,
                                      in_permutations,
                                      in_permutations_inverse,
                                      weight,
                                      mask,
                                      self.out_mapping,
                                      self.out_mapping_reverse,
                                      out_permutations,
                                      out_permutations_inverse
                                      )

    @staticmethod
    def permutation_mix(permutation_scores,
                        permutations,
                        temperature: float,
                        training: bool):
        if training: # True
            s = F.softmax(permutation_scores * temperature, dim=-1)
        else:
            s = torch.argmax(permutation_scores, dim=-1)
            s = F.one_hot(s, num_classes=permutation_scores.shape[-1]).float()

        s = s.matmul(permutations.reshape(permutations.shape[0], -1))
        s = s.view(-1, *permutations.shape[1:])

        return s

    @staticmethod
    def all_permutations(d_group):
        t = torch.tensor(list(itertools.permutations(range(d_group))))
        tp = t.sort(dim=1)[1]
        a = torch.arange(t.shape[0]).unsqueeze(-1).expand_as(t)
        c = torch.arange(d_group).unsqueeze(0).expand_as(t)

        ones = torch.stack([a, c, t], dim=-1).reshape(-1, 3).t()
        m = torch.zeros(t.shape[0], d_group, d_group)
        m[tuple(ones)] = 1.0

        ones = torch.stack([a, c, tp], dim=-1).reshape(-1, 3).t()
        mp = torch.zeros(t.shape[0], d_group, d_group)
        mp[tuple(ones)] = 1.0

        return t, tp, m, mp

    def random_permutation(iterable, r=None):
        "Random selection from itertools.permutations(iterable, r)"
        pool = tuple(iterable)
        r = len(pool) if r is None else r
        return tuple(random.sample(pool, r))

    @staticmethod
    def dimension_mapping(d, testing=False):
        while True:
            m = torch.tensor(DimensionShuffler.random_permutation(range(d)))
            if testing and (m == torch.arange(d)).all():
                continue
            return m

    @staticmethod
    def sequence_batch_group_permutation(s, mapping, permutations, final=False):
        d_group = permutations.shape[-1]
        d = s.shape[-1]
        assert ((d % d_group) == 0)
        assert (len(s.shape) == 3)
        s_shape = s.shape
        if not final:
            s = s[:, :, mapping]
        s = s.reshape(s.shape[:-1] + (s.shape[-1] // d_group, d_group))
        s2 = torch.einsum('ijmk,mkn->ijmn', s, permutations)
        s2 = s2.reshape(s_shape)
        if final:
            s2 = s2[:, :, mapping]
        return s2

    @staticmethod
    def matrix_group_permutation_inverse(matrix, mapping, permutations, permutations_inverse, transposed=False):
        d_group = permutations.shape[-1]
        d = matrix.shape[-1]
        assert ((d % d_group) == 0)
        assert (len(matrix.shape) == 2)
        matrix_shape = matrix.shape
        matrix = matrix[:, mapping]
        matrix = matrix.reshape(matrix.shape[0], matrix.shape[1] // d_group, d_group)
        permutations_m = permutations_inverse
        # mnk because matrix is transposed, we should transpose permutations_m too
        perm_selector = "mkn" if transposed else "mnk"
        matrix2 = torch.einsum(f'imk,{perm_selector}->imn', matrix, permutations_m)
        matrix2 = matrix2.reshape(matrix_shape)
        return matrix2

    @staticmethod
    def permutated_linear(s, in_map, in_permut, in_permut_inverse, matrix, mask, out_map, out_map_inverse, out_permut, out_permut_inverse):
        s_in = DimensionShuffler.sequence_batch_group_permutation(s, in_map, in_permut)
        matrix2 = DimensionShuffler.matrix_group_permutation_inverse(matrix, in_map, in_permut, in_permut_inverse)
        matrix3 = DimensionShuffler.matrix_group_permutation_inverse(matrix2.t(), out_map, out_permut, out_permut_inverse, transposed=True)
        matrix3 = matrix3 * mask.t()
        s_inner = s_in.matmul(matrix3)
        s_out = DimensionShuffler.sequence_batch_group_permutation(s_inner, out_map_inverse, out_permut, final=True)

        return s_out

        s_ref = s.matmul(matrix.t()) # REFERENCE

        max_std = (s_out - s_ref).std().item()
        max_diff = (s_out - s_ref).abs().max().item()
        if max_diff > 0.1:
            print("max difference", max_diff)

        return s_out


class MaskDimensionShuffler(nn.Module):
    def __init__(self, in_features, out_features, in_features_group=4, out_features_group=4):
        super().__init__()
        self.in_features = in_features
        self.in_features_group = in_features_group
        self.out_features = out_features
        self.out_features_group = out_features_group

        in_mapping = self.dimension_mapping(in_features)
        in_mapping_reverse = in_mapping.sort()[1]
        out_mapping = self.dimension_mapping(out_features)
        out_mapping_reverse = out_mapping.sort()[1]

        self.register_buffer("in_mapping", in_mapping)
        self.register_buffer("in_mapping_reverse", in_mapping_reverse)
        self.register_buffer("out_mapping", out_mapping)
        self.register_buffer("out_mapping_reverse", out_mapping_reverse)

        if in_features_group == 2:
            score_dim = 1
        else:
            # Currently not supported
            assert (False)

        in_permutation_scores = torch.randn(in_features // in_features_group, score_dim)
        out_permutation_scores = torch.randn(out_features // out_features_group, score_dim)

        self.in_permutation_scores = nn.Parameter(in_permutation_scores)
        self.out_permutation_scores = nn.Parameter(out_permutation_scores)

    @staticmethod
    def random_permutation(iterable, r=None):
        "Random selection from itertools.permutations(iterable, r)"
        pool = tuple(iterable)
        r = len(pool) if r is None else r
        return tuple(random.sample(pool, r))

    @staticmethod
    def dimension_mapping(d):
        return torch.tensor(MaskDimensionShuffler.random_permutation(range(d)))

    @staticmethod
    def rotations_2d(angles):
        c = angles.cos()
        s = angles.sin()

        rot0 = torch.stack([c, -s], dim=-1)
        rot1 = torch.stack([s, c], dim=-1)
        rot = torch.stack([rot0, rot1], dim=1)
        return rot

    @staticmethod
    def angles(scores, temperature, training):
        scores_0 = torch.zeros_like(scores)
        scores = torch.stack([scores_0, scores], dim=-1)

        if training:
            s = F.softmax(scores * temperature, dim=-1)
        else:
            s = torch.argmax(scores, dim=-1)
            s = F.one_hot(s, num_classes=scores.shape[-1]).float()

        angles = s[:, :, 1] * (math.pi * 0.5)
        return angles

    @staticmethod
    def matrices(angles):
        if angles.shape[-1] == 1:
            return MaskDimensionShuffler.rotations_2d(angles.squeeze(-1))
        else:
            assert(False)

    @staticmethod
    def rotate(mask, mapping, mapping_reverse, scores, temperature, training):
        # Rotate each group of n lines
        mask_shape = mask.shape
        # Get the rotations angles
        angles0 = MaskDimensionShuffler.angles(scores, temperature, training)
        mat0 = MaskDimensionShuffler.matrices(angles0)

        # The mixing factors are actually the squares of each of the coefficient of the rotation matrix
        mat0 = mat0 * mat0

        # Apply the global random dimension remapping
        if mapping is not None:
            mask = StaticIndexDim1.apply(mask, mapping, mapping_reverse)

        # Create the groups of dimensions
        rot_dim = mat0.shape[-1]
        mask = mask.view(mask_shape[0], mask_shape[1] // rot_dim, rot_dim)

        # Following lines: Rotate each group: we could use an einsum, but not much more difficult to do
        # Adapt the mask to the shape of matrices by repeating the last dimension
        mask = mask.unsqueeze(-1).repeat(1, 1, 1, rot_dim)
        # Adapt the matrices to the shape of the mask
        mat0 = mat0.unsqueeze(0).repeat(mask.shape[0], 1, 1, 1)

        # Finish with the sum on the right dimension
        mask = (mat0 * mask).sum(-2)

        # Reshape the mask to remove the temporary grouping
        return mask.view(mask_shape)

    @staticmethod
    def final_mapping(mapping, scores):
        mapping = mapping.view(1, mapping.shape[0])
        mapping = MaskDimensionShuffler.rotate(mapping, None, None, scores, 0, False)
        mapping = (mapping.round() + 0.25).long().squeeze(0)

        return mapping

    def final_mappings(self):
        # Those are the mappings that should be applied to the weights
        # (and so inverted)
        m0 = self.final_mapping(self.in_mapping, self.in_permutation_scores)
        m0_p = m0.sort()[1]
        m1 = self.final_mapping(self.out_mapping, self.out_permutation_scores)
        m1_p = m1.sort()[1]
        return m0, m0_p, m1, m1_p

    def forward(self, mask, temperature):
        training = self.training
        mask = self.rotate(mask, self.in_mapping, self.in_mapping_reverse, self.in_permutation_scores, temperature, training)
        mask = mask.t()
        mask = self.rotate(mask, self.out_mapping, self.out_mapping_reverse, self.out_permutation_scores, temperature, training)
        mask = mask.t()
        return mask


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
        shuffling_method:str = "disabled",
        in_shuffling_group:int = 4,
        out_shuffling_group:int = 4,
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
        super().__init__(in_features=in_features, out_features=out_features, bias=bias)
        assert pruning_method in ["topK", "threshold", "sigmoied_threshold", "magnitude", "l0"]
        self.pruning_method = pruning_method
        self.mask_block_rows = mask_block_rows
        self.mask_block_cols = mask_block_cols
        AMPERE_METHODS = ["disabled", "annealing"]
        if ampere_pruning_method not in AMPERE_METHODS:
            raise RuntimeError(f"Unknown ampere pruning method '{ampere_pruning_method}', should be in {AMPERE_METHODS}")
        self.ampere_pruning_method = ampere_pruning_method

        SHUFFLING_METHODS = ["disabled", "annealing", "mask_annealing"]
        if shuffling_method not in SHUFFLING_METHODS:
            raise RuntimeError(f"Unknown shuffle method '{shuffling_method}', should be in {SHUFFLING_METHODS}")

        self.shuffling_method = shuffling_method
        assert in_shuffling_group >= 1
        self.in_shuffling_group = in_shuffling_group
        assert out_shuffling_group >= 1
        self.out_shuffling_group = out_shuffling_group

        self.shuffler = None
        self.mask_shuffler = None

        if self.shuffling_method == "annealing":
            self.shuffler = DimensionShuffler(in_features=in_features,
                                              out_features=out_features,
                                              in_features_group=self.in_shuffling_group,
                                              out_features_group=self.out_shuffling_group)
        elif self.shuffling_method == "mask_annealing":
            self.mask_shuffler = MaskDimensionShuffler(in_features=in_features,
                                                       out_features=out_features,
                                                       in_features_group=self.in_shuffling_group,
                                                       out_features_group=self.out_shuffling_group)

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
            self.initialize_ampere_permut_scores()
        else:
            self.register_parameter("ampere_permut_scores", None)


    def initialize_ampere_permut_scores(self):
        """"We must remember that weights are used in transposed form for forward pass,
        which we want to optimize the most.
        So we make sure we are creating an Ampere sparse pattern on the right dimension -> 0"""
        assert ((self.weight.shape[0] % AMPERE_M) == 0)

        sparse_patterns_count = ampere_pattern(None).shape[0]
        # Creating the pattern in a transposed way to avoid a few ops later
        ampere_mask_size = (self.weight.shape[1], self.weight.shape[0] // AMPERE_M, sparse_patterns_count)
        self.ampere_permut_scores = nn.Parameter(torch.Tensor(size=ampere_mask_size))

        if self.ampere_mask_init == "constant":
            init.constant_(self.ampere_permut_scores, val=self.ampere_mask_scale)
        elif self.ampere_mask_init == "uniform":
            init.uniform_(self.ampere_permut_scores, a=-self.ampere_mask_scale, b=self.ampere_mask_scale)
        elif self.ampere_mask_init == "kaiming":
            init.kaiming_uniform_(self.ampere_permut_scores, a=math.sqrt(5))

    def init_mask(self):
        if self.mask_init == "constant":
            init.constant_(self.mask_scores, val=self.mask_scale)
        elif self.mask_init == "uniform":
            init.uniform_(self.mask_scores, a=-self.mask_scale, b=self.mask_scale)
        elif self.mask_init == "kaiming":
            init.kaiming_uniform_(self.mask_scores, a=math.sqrt(5))

    @staticmethod
    def expand_mask_(mask, mask_block_rows, mask_block_cols):
        mask = torch.repeat_interleave(mask, mask_block_rows, dim=0)
        mask = torch.repeat_interleave(mask, mask_block_cols, dim=1)
        return mask

    @staticmethod
    def ampere_mask_(ampere_permut_scores,
                     ampere_temperature: float,
                     device:torch.DeviceObjType,
                     training:bool):
        if training:
            s = F.softmax(ampere_permut_scores * ampere_temperature, dim=-1)
        else:
            s = torch.argmax(ampere_permut_scores, dim=-1)
            s = F.one_hot(s, num_classes=ampere_permut_scores.shape[-1]).float()

        s = s.matmul(ampere_pattern(device))
        s = s.view(-1, s.shape[1] * s.shape[2])
        s = s.t()

        return s

    @staticmethod
    def check_name(name):
        return name.endswith(".ampere_permut_scores") or name.endswith(".mask_scores")

    @staticmethod
    def mask_(weight,
                       pruning_method,
                       threshold,
                       mask_scores,
                       ampere_pruning_method,
                       ampere_temperature,
                       ampere_permut_scores,
                       mask_block_rows,
                       mask_block_cols,
                       training):
        if pruning_method == "topK":
            mask = TopKBinarizer.apply(mask_scores, threshold)
        elif pruning_method in ["threshold", "sigmoied_threshold"]:
            sig = "sigmoied" in pruning_method
            mask = ThresholdBinarizer.apply(mask_scores, threshold, sig)
        elif pruning_method == "magnitude":
            mask = MagnitudeBinarizer.apply(weight, threshold)
        elif pruning_method == "l0":
            l, r, b = -0.1, 1.1, 2 / 3
            if training:
                u = torch.zeros_like(mask_scores).uniform_().clamp(0.0001, 0.9999)
                s = torch.sigmoid((u.log() - (1 - u).log() + mask_scores) / b)
            else:
                s = torch.sigmoid(mask_scores)
            s_bar = s * (r - l) + l
            mask = s_bar.clamp(min=0.0, max=1.0)
        # Expand block mask to individual element mask
        if pruning_method != "magnitude":
            mask = MaskedLinear.expand_mask_(mask,
                                             mask_block_rows=mask_block_rows,
                                             mask_block_cols=mask_block_cols
                                             )

        if ampere_pruning_method != "disabled":
            ampere_mask = MaskedLinear.ampere_mask_(ampere_permut_scores,
                                                    ampere_temperature,
                                                    device=mask.device,
                                                    training=training)
            mask = mask * ampere_mask

        return mask

    @staticmethod
    def masked_weights_from_state_dict(state_dict,
                                       weight_name,
                                       pruning_method,
                                       threshold,
                                       ampere_pruning_method,
                                       mask_block_rows,
                                       mask_block_cols):
        def name_for_mask(weight_name, mask_name):
            new_name = weight_name.split(".")[:-1] + [mask_name]
            new_name = ".".join(new_name)

        parameters = {}
        for name in ["weight", "mask_scores", "ampere_permut_scores"]:
            parameters[name] = state_dict.get(name_for_mask(weight_name, name))
            
        ret = MaskedLinear.masked_weights(pruning_method=pruning_method,
                                          threshold=threshold,
                                          ampere_pruning_method=ampere_pruning_method,
                                          ampere_temperature=0.0,
                                          training=False,
                                          mask_block_rows=mask_block_rows,
                                          mask_block_cols=mask_block_cols,
                                          **parameters)
                                          
        return ret

    def expand_mask(self, mask):
        return self.expand_mask_(mask, self.mask_block_rows, self.mask_block_cols)


    def forward(self, input: torch.tensor, current_config: dict):
        # Get the mask
        threshold = current_config["threshold"]
        ampere_temperature = current_config["ampere_temperature"]
        shuffle_temperature = current_config["shuffling_temperature"]

        mask = self.mask_(self.weight,
                          self.pruning_method,
                          threshold,
                          self.mask_scores,
                          self.ampere_pruning_method,
                          ampere_temperature,
                          self.ampere_permut_scores,
                          self.mask_block_rows,
                          self.mask_block_cols,
                          training=self.training)

        if self.shuffler is not None:
            return self.shuffler(input, self.weight, mask, shuffle_temperature) + self.bias
        else:
            if self.mask_shuffler is not None:
                mask = self.mask_shuffler(mask, shuffle_temperature)
            weight_thresholded = mask * self.weight
            # Compute output (linear layer) with masked weights
            return F.linear(input, weight_thresholded, self.bias)
