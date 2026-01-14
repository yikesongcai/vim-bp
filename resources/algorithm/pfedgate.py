import os
from collections import OrderedDict

import torch.nn.functional as F
import torch
from copy import copy, deepcopy
import string
import itertools
# from numba import jit
from torch.nn.functional import gumbel_softmax
import math
import torch
import torch.nn.functional as F
import torch.nn
import torch.nn as nn
import numpy as np
import flgo.algorithm.fedavg as fedavg
import flgo.utils.fmodule as fuf
from torch.utils.data import Dataset
from torchvision.transforms import RandomCrop, RandomHorizontalFlip
import torchvision
import torchvision.models.resnet as rn
import flgo.utils.submodule as usub

MIN_SPARSE_FACTOR = 0.2

class Server(fedavg.Server):
    def initialize(self, *args, **kwargs):
        self.init_algo_para({'eta':0.1,'fine_grained_block_split':5, 'max_block_split':100})
        min_cap = min([c._capacity for c in self.clients])
        self.running_stats_for_gating_weight = RunningStats()
        self.model = self._model_class.init_global_module(self)

    def pack(self, client_id, mtype=0, *args, **kwargs):
        return {"model": deepcopy(self.model.model)}

    def iterate(self):
        self.selected_clients = self.sample()
        models = self.communicate(self.selected_clients)['model']
        self.model.model = self.aggregate(models)
        return

class Client(fedavg.Client):
    def initialize(self, *args, **kwargs):
        self.model = self._model_class.init_local_module(self)
        self.p = 1.0

    def unpack(self, received_pkg):
        self.model.model = received_pkg['model']
        return self.model

    def pack(self, model, *args, **kwargs):
        return {'model': self.model.model}

    @fuf.with_multi_gpus
    def train(self, model):
        model.train()
        optimizer = torch.optim.SGD(
            [{'params': [p for p in model.model.parameters()], 'lr':self.learning_rate}, {'params': [p for p in model.gating_layer.parameters()], 'lr':self.eta}],  weight_decay=self.weight_decay, momentum=self.momentum)
        for iter in range(self.num_steps):
            # get a batch of data
            batch_data = self.get_batch_data()
            model.zero_grad()
            # calculate the loss of the model on batched dataset through task-specified calculator
            loss = self.calculator.compute_loss(model, batch_data)['loss']
            loss.backward()
            if self.clip_grad>0:torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=self.clip_grad)
            optimizer.step()
        return

def init_global_module(object):
    module_class = eval(os.path.split(object.option['task'])[-1].upper().split('_')[0]+'Model')
    return module_class.init_global_module(object)

def init_local_module(object):
    module_class = eval(os.path.split(object.option['task'])[-1].upper().split('_')[0]+'Model')
    return module_class.init_local_module(object)

def init_dataset(object):
    module_class = eval(os.path.split(object.option['task'])[-1].upper().split('_')[0]+'Model')
    return module_class.init_dataset(object)

def deepgetattr(obj, attr):
    ""
    "Recurses through an attribute chain to get the ultimate value."
    ""
    for sub_attr in attr.split("."):
        if sub_attr[0] == "[" and sub_attr[-1] == "]":
            obj = obj[sub_attr[1:-1]]
        else:
            obj = getattr(obj, sub_attr)
    # return reduce(getattr, attr.split('.'), obj)
    return obj

def map_module_name(name):
    mapped_name = [f"_modules.[{m_name}]" if m_name.isnumeric() else m_name for m_name in name.split(".")]
    return ".".join(mapped_name)

class SwitchNorm1d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.997, using_moving_average=True):
        super(SwitchNorm1d, self).__init__()
        self.eps = eps
        self.momentum = momentum
        self.using_moving_average = using_moving_average
        self.weight = nn.Parameter(torch.ones(1, num_features))
        self.bias = nn.Parameter(torch.zeros(1, num_features))
        self.mean_weight = nn.Parameter(torch.ones(2))
        self.var_weight = nn.Parameter(torch.ones(2))
        self.register_buffer('running_mean', torch.zeros(1, num_features))
        self.register_buffer('running_var', torch.zeros(1, num_features))
        self.reset_parameters()

    def reset_parameters(self):
        self.running_mean.zero_()
        self.running_var.zero_()
        self.weight.data.fill_(1)
        self.bias.data.zero_()

    def _check_input_dim(self, input):
        if input.dim() != 2:
            raise ValueError('expected 2D input (got {}D input)'
                             .format(input.dim()))

    def forward(self, x):
        self._check_input_dim(x)
        mean_ln = x.mean(1, keepdim=True)
        var_ln = x.var(1, keepdim=True)

        if self.training:
            mean_bn = x.mean(0, keepdim=True)
            var_bn = x.var(0, keepdim=True)
            if self.using_moving_average:
                self.running_mean.mul_(self.momentum)
                self.running_mean.add_((1 - self.momentum) * mean_bn.data)
                self.running_var.mul_(self.momentum)
                self.running_var.add_((1 - self.momentum) * var_bn.data)
            else:
                self.running_mean.add_(mean_bn.data)
                self.running_var.add_(mean_bn.data ** 2 + var_bn.data)
        else:
            mean_bn = torch.autograd.Variable(self.running_mean)
            var_bn = torch.autograd.Variable(self.running_var)

        softmax = nn.Softmax(0)
        mean_weight = softmax(self.mean_weight)
        var_weight = softmax(self.var_weight)

        mean = mean_weight[0] * mean_ln + mean_weight[1] * mean_bn
        var = var_weight[0] * var_ln + var_weight[1] * var_bn

        x = (x - mean) / (var + self.eps).sqrt()
        return x * self.weight + self.bias

class SwitchNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.9, using_moving_average=True, using_bn=True,
                 last_gamma=False):
        super(SwitchNorm2d, self).__init__()
        self.eps = eps
        self.momentum = momentum
        self.using_moving_average = using_moving_average
        self.using_bn = using_bn
        self.last_gamma = last_gamma
        self.weight = nn.Parameter(torch.ones(1, num_features, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, num_features, 1, 1))
        if self.using_bn:
            self.mean_weight = nn.Parameter(torch.ones(3))
            self.var_weight = nn.Parameter(torch.ones(3))
        else:
            self.mean_weight = nn.Parameter(torch.ones(2))
            self.var_weight = nn.Parameter(torch.ones(2))
        if self.using_bn:
            self.register_buffer('running_mean', torch.zeros(1, num_features, 1))
            self.register_buffer('running_var', torch.zeros(1, num_features, 1))

        self.reset_parameters()

    def reset_parameters(self):
        if self.using_bn:
            self.running_mean.zero_()
            self.running_var.zero_()
        if self.last_gamma:
            self.weight.data.fill_(0)
        else:
            self.weight.data.fill_(1)
        self.bias.data.zero_()

    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'
                             .format(input.dim()))

    def forward(self, x):
        self._check_input_dim(x)
        N, C, H, W = x.size()
        x = x.view(N, C, -1)
        mean_in = x.mean(-1, keepdim=True)
        var_in = x.var(-1, keepdim=True)

        mean_ln = mean_in.mean(1, keepdim=True)
        temp = var_in + mean_in ** 2
        var_ln = temp.mean(1, keepdim=True) - mean_ln ** 2

        if self.using_bn:
            if self.training:
                mean_bn = mean_in.mean(0, keepdim=True)
                var_bn = temp.mean(0, keepdim=True) - mean_bn ** 2
                if self.using_moving_average:
                    self.running_mean.mul_(self.momentum)
                    self.running_mean.add_((1 - self.momentum) * mean_bn.data)
                    self.running_var.mul_(self.momentum)
                    self.running_var.add_((1 - self.momentum) * var_bn.data)
                else:
                    self.running_mean.add_(mean_bn.data)
                    self.running_var.add_(mean_bn.data ** 2 + var_bn.data)
            else:
                mean_bn = torch.autograd.Variable(self.running_mean)
                var_bn = torch.autograd.Variable(self.running_var)

        softmax = nn.Softmax(0)
        mean_weight = softmax(self.mean_weight)
        var_weight = softmax(self.var_weight)

        if self.using_bn:
            mean = mean_weight[0] * mean_in + mean_weight[1] * mean_ln + mean_weight[2] * mean_bn
            var = var_weight[0] * var_in + var_weight[1] * var_ln + var_weight[2] * var_bn
        else:
            mean = mean_weight[0] * mean_in + mean_weight[1] * mean_ln
            var = var_weight[0] * var_in + var_weight[1] * var_ln

        x = (x - mean) / (var + self.eps).sqrt()
        x = x.view(N, C, H, W)
        return x * self.weight + self.bias

class SwitchNorm3d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.997, using_moving_average=True, using_bn=True,
                 last_gamma=False):
        super(SwitchNorm3d, self).__init__()
        self.eps = eps
        self.momentum = momentum
        self.using_moving_average = using_moving_average
        self.using_bn = using_bn
        self.last_gamma = last_gamma
        self.weight = nn.Parameter(torch.ones(1, num_features, 1, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, num_features, 1, 1, 1))
        if self.using_bn:
            self.mean_weight = nn.Parameter(torch.ones(3))
            self.var_weight = nn.Parameter(torch.ones(3))
        else:
            self.mean_weight = nn.Parameter(torch.ones(2))
            self.var_weight = nn.Parameter(torch.ones(2))
        if self.using_bn:
            self.register_buffer('running_mean', torch.zeros(1, num_features, 1))
            self.register_buffer('running_var', torch.zeros(1, num_features, 1))

        self.reset_parameters()

    def reset_parameters(self):
        if self.using_bn:
            self.running_mean.zero_()
            self.running_var.zero_()
        if self.last_gamma:
            self.weight.data.fill_(0)
        else:
            self.weight.data.fill_(1)
        self.bias.data.zero_()

    def _check_input_dim(self, input):
        if input.dim() != 5:
            raise ValueError('expected 5D input (got {}D input)'
                             .format(input.dim()))

    def forward(self, x):
        self._check_input_dim(x)
        N, C, D, H, W = x.size()
        x = x.view(N, C, -1)
        mean_in = x.mean(-1, keepdim=True)
        var_in = x.var(-1, keepdim=True)

        mean_ln = mean_in.mean(1, keepdim=True)
        temp = var_in + mean_in ** 2
        var_ln = temp.mean(1, keepdim=True) - mean_ln ** 2

        if self.using_bn:
            if self.training:
                mean_bn = mean_in.mean(0, keepdim=True)
                var_bn = temp.mean(0, keepdim=True) - mean_bn ** 2
                if self.using_moving_average:
                    self.running_mean.mul_(self.momentum)
                    self.running_mean.add_((1 - self.momentum) * mean_bn.data)
                    self.running_var.mul_(self.momentum)
                    self.running_var.add_((1 - self.momentum) * var_bn.data)
                else:
                    self.running_mean.add_(mean_bn.data)
                    self.running_var.add_(mean_bn.data ** 2 + var_bn.data)
            else:
                mean_bn = torch.autograd.Variable(self.running_mean)
                var_bn = torch.autograd.Variable(self.running_var)

        softmax = nn.Softmax(0)
        mean_weight = softmax(self.mean_weight)
        var_weight = softmax(self.var_weight)

        if self.using_bn:
            mean = mean_weight[0] * mean_in + mean_weight[1] * mean_ln + mean_weight[2] * mean_bn
            var = var_weight[0] * var_in + var_weight[1] * var_ln + var_weight[2] * var_bn
        else:
            mean = mean_weight[0] * mean_in + mean_weight[1] * mean_ln
            var = var_weight[0] * var_in + var_weight[1] * var_ln

        x = (x - mean) / (var + self.eps).sqrt()
        x = x.view(N, C, D, H, W)
        return x * self.weight + self.bias

class DifferentiableRoundFun(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.input = input
        return torch.round(input).int()

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input

class DifferentiableCeilFun(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.input = input
        return torch.ceil(input)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input

class AdaptedLinear(nn.Linear):

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(AdaptedLinear, self).__init__(in_features, out_features, bias, device, dtype)
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

        self.weight_adapted = None
        self.bias_adapted = None
        self.forward_with_adapt_para = False

    def forward(self, input):
        if self.forward_with_adapt_para:
            weight = self.weight_adapted
            bias = self.bias_adapted
        else:
            weight = self.weight
            bias = self.bias
        return F.linear(input, weight, bias)

class Reshape(nn.Module):
    def __init__(self, reshape_type="flat_dim0"):
        super(Reshape, self).__init__()
        self.reshape_type = reshape_type

    def forward(self, x):
        if self.reshape_type == "flat_dim0":
            B = x.size()[0]
            return x.view(B, -1)
        else:
            raise NotImplementedError("Un-supported reshape_type: {}".format(self.reshape_type))

class GatingLayer(nn.Module):
    IN_PLANES_TYPE = {
        "cifar10": 3,
        "cifar100": 3,
        "fashion": 1,
        "mnist": 1,
        "femnist": 1,
    }
    def __init__(self, model_to_mask, dataset_name, fine_grained_block_split=100, seperate_trans=0):
        super().__init__()
        assert dataset_name in ["shakespeare", "emnist", "femnist", "cifar10", "cifar100", "fashion", "mnist"], \
            f"Un-supported dataset name: {dataset_name}"

        # ----------------------------  Split Blocks into linear and non-linear parts  ----------------------------
        # model-block size, used for structured masking
        ori_block_names = [p[0] for p in model_to_mask.named_parameters()]
        ori_block_size_lookup_table = torch.tensor([p.numel() for p in model_to_mask.parameters()])
        # linear_layer_idx is used to select top masks for linear_layer and non-linear layer separately,
        # since linear layers usually have much larger parameters than other types
        linear_layer_block_idx = []
        for i, p in enumerate(model_to_mask.named_parameters()):
            para_name_second_last_level = ".".join(p[0].split(".")[:-1])
            para_name_second_last_level = map_module_name(para_name_second_last_level)
            para = deepgetattr(model_to_mask, para_name_second_last_level)
            if isinstance(para, torch.nn.Linear) or isinstance(para, AdaptedLinear):
                linear_layer_block_idx.append(i)

        self.fine_grained_block_split = fine_grained_block_split
        # a dict stores "para_name" -> (para_sub_block_begin_idx, para_sub_block_end_idx, size_each_sub)
        self.para_name_to_block_split_info = {}
        assert self.fine_grained_block_split >= 1, f"fine_grained_block_split must >= 1, while got {fine_grained_block_split}"
        if self.fine_grained_block_split == 1:
            self.block_names = ori_block_names
            self.block_size_lookup_table = ori_block_size_lookup_table
            self.linear_layer_block_idx = linear_layer_block_idx
        else:
            # each block is further split into fine_grained_block_split parts
            self.block_names = []
            self.block_size_lookup_table = []
            self.linear_layer_block_idx = []
            cur_block_idx = 0
            for i, name in enumerate(ori_block_names):
                # e.g., block_size = 25, fine_grained_block_split = 10
                split_num = self.fine_grained_block_split \
                    if ori_block_size_lookup_table[i] >= self.fine_grained_block_split \
                    else ori_block_size_lookup_table[i]
                # e.g., size_each_sub = ceil(25/10) = 3
                size_each_sub = torch.ceil(ori_block_size_lookup_table[i] / split_num).int()
                # e.g., split_num = ceil(25/3) = 9
                split_num = torch.ceil(ori_block_size_lookup_table[i] / size_each_sub).int()
                self.para_name_to_block_split_info[name] = (cur_block_idx, cur_block_idx + split_num, size_each_sub)
                for j in range(split_num - 1):
                    self.block_names.append(f"{name}.sub{j}")
                    self.block_size_lookup_table.append(size_each_sub)
                    if i in linear_layer_block_idx:
                        self.linear_layer_block_idx.append(cur_block_idx)
                    cur_block_idx += 1
                # for the last sub-block, in the case: (ori_block_size_lookup_table[i] % fine_grained_block_split !=0)
                self.block_names.append(f"{name}.sub{split_num - 1}")
                # e.g., size_last_sub = 25 - (9-1)*3 = 1
                size_last_sub = ori_block_size_lookup_table[i] - (split_num - 1) * size_each_sub
                self.block_size_lookup_table.append(size_last_sub)
                if i in linear_layer_block_idx:
                    self.linear_layer_block_idx.append(cur_block_idx)
                cur_block_idx += 1
        self.non_linear_layer_block_idx = [i for i in range(len(self.block_size_lookup_table))
                                           if i not in self.linear_layer_block_idx]

        # filter out the idx of first sub-blocks for each components,
        # to avoid cutting info flow (some internal sub-blocks are all zeros)
        self.linear_layer_block_idx_filter_first = []
        for idx in self.linear_layer_block_idx:
            if "sub0" not in self.block_names[idx]:
                self.linear_layer_block_idx_filter_first.append(idx)
        self.non_linear_layer_block_idx_filter_first = []
        for idx in self.non_linear_layer_block_idx:
            if "sub0" not in self.block_names[idx]:
                self.non_linear_layer_block_idx_filter_first.append(idx)

        if self.fine_grained_block_split != 1:
            self.block_size_lookup_table = torch.stack(self.block_size_lookup_table)

        block_size_lookup_table_linear = self.block_size_lookup_table[self.linear_layer_block_idx].double()
        block_size_lookup_table_non_linear = self.block_size_lookup_table[self.non_linear_layer_block_idx].double()
        block_size_lookup_table_linear /= block_size_lookup_table_linear.sum()
        block_size_lookup_table_non_linear /= block_size_lookup_table_non_linear.sum()
        self.block_size_lookup_table_normalized = deepcopy(self.block_size_lookup_table).double()
        self.block_size_lookup_table_normalized[self.linear_layer_block_idx] = block_size_lookup_table_linear
        self.block_size_lookup_table_normalized[self.non_linear_layer_block_idx] = block_size_lookup_table_non_linear

        # ---------------------------- Build gating layer ----------------------------
        if dataset_name == "shakespeare":
            norm_input_layer = SwitchNorm1d(self.IN_PLANES_TYPE[dataset_name])
            raise RuntimeError("Did not support Shakespeare")
            # input_feat_size = SHAKESPEARE_CONFIG["embed_size"]
        else:
            norm_input_layer = SwitchNorm2d(self.IN_PLANES_TYPE[dataset_name])
            input_feat_size = 1 * 28 * 28 if dataset_name in ["emnist", "femnist", "mnist", "fashion"] else 3 * 32 * 32
        reshape_layer = Reshape(reshape_type="flat_dim0")
        output_layer = nn.Linear(in_features=input_feat_size, out_features=len(self.block_names))
        norm_outputs = torch.nn.BatchNorm1d(num_features=len(self.block_names))
        self.norm_input_layer = norm_input_layer
        self.gating = nn.Sequential(
            reshape_layer,
            output_layer,
            norm_outputs
        )

        self.seperate_trans = seperate_trans
        if self.seperate_trans == 1:
            self.w_transform = deepcopy(self.gating)
            self.w_transform
        self.norm_input_layer
        self.gating

        self.norm_input_each_forward = True

    def forward(self, x):
        assert len(x.size()) in [3, 4], f"Un-expected input shape for gating layer, got {len(x.size())}"
        if self.norm_input_each_forward:
            x = self.norm_input_layer(x)
        gating_score = self._forward(x, self.gating)
        if self.seperate_trans == 1:
            trans_weight = self._forward(x, self.w_transform)
        else:
            trans_weight = gating_score
        return gating_score, trans_weight

    def _forward(self, x, layer):
        # predict suitable sub-blocks of base-model according to given example
        res = layer(x)
        if len(res.size()) == 3:  # [B, seq_len, block_len] for texts
            return torch.mean(res, dim=1)
        elif len(res.size()) == 2:  # [B, block_len] for images
            return res
        else:
            raise RuntimeError(f"Un-expected mask weights shape for gating layer, got {len(res.size())}")

class KnapsackSolver01(object):
    """
    A knapsack problem solver implementation for 0-1 Knapsack with large Weights,
    ref: https://www.geeksforgeeks.org/knapsack-with-large-weights/

    time complexity: O(value_sum_max * item_num_max) = O(item_num_max * item_num_max) in our setting
    auxiliary space: O(value_sum_max * item_num_max) = O(item_num_max * item_num_max) in our setting
    """

    def __init__(self, value_sum_max, item_num_max, weight_max):
        self.value_sum_max = value_sum_max
        self.item_num_max = item_num_max
        self.weight_max = weight_max

        # dp[V][i] represents the minimum weight subset of the subarray arr[i,...,N-1]
        # required to get a value of at least V.
        self.dp = np.zeros((value_sum_max + 1, item_num_max))
        self.value_solved = np.zeros((value_sum_max + 1, item_num_max))
        self.selected_item = np.zeros(item_num_max)

    def reset_state(self, value_sum_max=0, item_num_max=0, weight_max=0, iter_version=False):
        self.value_sum_max = self.value_sum_max if value_sum_max == 0 else value_sum_max
        self.item_num_max = self.item_num_max if item_num_max == 0 else item_num_max
        self.weight_max = self.weight_max if weight_max == 0 else weight_max
        self.selected_item = np.zeros(item_num_max)

        if not iter_version:
            self.value_solved = np.zeros((self.value_sum_max + 1, self.item_num_max))
            self.dp = np.zeros((self.value_sum_max + 1, self.item_num_max))
        else:
            self.dp = np.full((self.value_sum_max + 1, self.item_num_max), -1)
            self.selected_item = np.zeros((self.value_sum_max + 1, self.item_num_max))

    # Function to solve the recurrence relation
    def solve_dp(self, r, i, w, val, n):
        # Base cases
        if r <= 0:
            return 0
        if i == n:
            return self.weight_max
        if self.value_solved[r][i]:
            return self.dp[r][i]

        # Marking state as solved
        self.value_solved[r][i] = 1

        # Recurrence relation.  the maximum recursive depth is n
        # self.dp[r][i] = min(self.solve_dp(r, i + 1, w, val, n),
        #                     w[i] + self.solve_dp(r - val[i], i + 1, w, val, n))
        w_discard_item_i = self.solve_dp(r, i + 1, w, val, n)
        # r - val[i] indicates the value must be from item i, i.e., select the item
        w_hold_item_i = w[i] + self.solve_dp(r - val[i], i + 1, w, val, n)

        if w_discard_item_i < w_hold_item_i:
            self.dp[r][i] = w_discard_item_i
            self.selected_item[i] = 0
        else:
            self.dp[r][i] = w_hold_item_i
            self.selected_item[i] = 1

        return self.dp[r][i]

    def found_max_value(self, weight_list, value_list, capacity):
        value_sum_max = int(sum(value_list))
        weight_max = int(sum(weight_list))
        # weight_max = capacity  # the 86.7 version, while always select the last ones
        self.reset_state(value_sum_max=value_sum_max, item_num_max=len(weight_list), weight_max=weight_max)
        # Iterating through all possible values
        # to find the the largest value that can
        # be represented by the given weights and capacity constraints
        for i in range(value_sum_max, -1, -1):
            res = self.solve_dp(i, 0, weight_list, value_list, self.item_num_max)
            if res <= capacity:
                return i, np.nonzero(self.selected_item)

        return 0, np.nonzero(self.selected_item)

    def found_max_value_greedy(self, weight_list, value_list, capacity):
        if isinstance(value_list, np.ndarray):
            sorted_idx = (-value_list).argsort()[:len(value_list)]
        elif isinstance(value_list, torch.Tensor):
            sorted_value_per_weight, sorted_idx = value_list.sort(descending=True)
        else:
            raise NotImplementedError(
                f"found_max_value_greedy, only support value_list as ndarray or Tensor, while got {type(value_list)}")
        selected_idx = []
        droped_idx = []
        total_weight = 0
        total_value = 0
        for idx in sorted_idx:
            weight_after_select = total_weight + weight_list[idx]
            if weight_after_select <= capacity:
                total_weight = weight_after_select
                total_value += value_list[idx]
                selected_idx.append(idx)
            else:
                droped_idx.append(idx)

        return total_value, total_weight, selected_idx, droped_idx

class ItemUnitCost:
    def __init__(self, wt, val, ind):
        self.wt = wt
        self.val = val
        self.ind = ind
        self.cost = val / wt

    def __lt__(self, other):
        return self.cost < other.cost

class KnapsackSolverFractional(object):
    """
    A knapsack problem solver implementation for fractional Knapsack,
    in which each item can be selected with a sub-part.
    ref: https://home.cse.ust.hk/~dekai/271/notes/L14/L14.pdf

    # Greedy Approach
    time complexity: O(N*logN) as the solution is based on sort
    """

    def __init__(self, item_num_max, weight_max):
        self.item_num_max = item_num_max
        self.weight_max = weight_max

        # stored the results
        self.selected_item = np.zeros(item_num_max)

    def reset_state(self, item_num_max, weight_max):
        self.item_num_max = item_num_max
        self.weight_max = weight_max
        self.selected_item = np.zeros(item_num_max)

    def _found_max_value_numeric(self, weight_list, value_list, capacity):
        self.selected_item = np.zeros(self.item_num_max)
        iVal = []
        for i in range(len(weight_list)):
            iVal.append(ItemUnitCost(weight_list[i], value_list[i], i))
        iVal.sort(reverse=True)
        totalValue = 0
        for item in iVal:
            curWt = item.wt
            curVal = item.val
            # select the whole item
            if capacity - curWt >= 0:
                capacity -= curWt
                self.selected_item[item.ind] = curWt
                totalValue += curVal
            # select sub-set of the item
            else:
                fraction = capacity / curWt
                self.selected_item[item.ind] = capacity
                totalValue += curVal * fraction
                capacity = capacity - (curWt * fraction)
                break
        return totalValue, self.selected_item, self.selected_item / np.array(weight_list)

    def _found_max_value_tensor(self, weight_list, value_list, capacity):
        self.selected_item = torch.zeros_like(weight_list, device=weight_list.device)

        value_per_weight = value_list / weight_list
        sorted_value_per_weight, sorted_idx = value_per_weight.sort(descending=True)

        total_val = 0
        for i, unit_value in enumerate(sorted_value_per_weight):
            cur_item_idx = sorted_idx[i]
            cur_weight = weight_list[cur_item_idx]
            cur_val = value_list[cur_item_idx]
            # select the whole item
            if capacity - cur_weight >= 0:
                capacity -= cur_weight
                self.selected_item[cur_item_idx] = cur_weight
                total_val += cur_val
            # select sub-set of the item
            else:
                fraction = capacity / cur_weight
                self.selected_item[cur_item_idx] = capacity
                total_val += cur_val * fraction
                capacity = capacity - (cur_weight * fraction)
                break
        return total_val, self.selected_item, self.selected_item / weight_list

    def found_max_value(self, weight_list, value_list, capacity):
        """function to get maximum value """
        self.reset_state(item_num_max=len(weight_list), weight_max=capacity)

        tensor_type = isinstance(weight_list, torch.Tensor) and isinstance(value_list, torch.Tensor)
        normal_numeric_type = isinstance(weight_list[0], (int, float, complex)) and isinstance(value_list[0],
                                                                                               (int, float, complex))
        assert tensor_type or normal_numeric_type, \
            f"Unsupported weight_list: {type(weight_list)}, value_list: {type(value_list)}"

        if tensor_type:
            res = self._found_max_value_tensor(weight_list, value_list, capacity)
        else:
            res = self._found_max_value_numeric(weight_list, value_list, capacity)
        return res

class RunningStats(object):

    def __init__(self):
        self.n = 0
        self.old_m = 0
        self.new_m = 0
        self.old_s = 0
        self.new_s = 0

    def reset(self):
        self.n = 0
        self.old_m = 0
        self.new_m = 0
        self.old_s = 0
        self.new_s = 0

    def clear(self):
        self.n = 0

    def push(self, x):
        self.n += 1

        if self.n == 1:
            self.old_m = self.new_m = x
            self.old_s = 0
        else:
            self.new_m = self.old_m + (x - self.old_m) / self.n
            self.new_s = self.old_s + (x - self.old_m) * (x - self.new_m)

            self.old_m = self.new_m
            self.old_s = self.new_s

    def mean(self):
        return self.new_m if self.n else 0.0

    def variance(self):
        return self.new_s / (self.n - 1) if self.n > 1 else 0.0

    def standard_deviation(self):
        if isinstance(self.new_s, torch.Tensor):
            return torch.sqrt(self.variance())
        else:
            return math.sqrt(self.variance())

class PGBlock(nn.Module):
    expansion: int = 1
    def __init__(
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
            downsample = None,
            groups: int = 1,
            base_width: int = 64,
            dilation: int = 1,
            norm_layer = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = rn.conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = rn.conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

    def adapted_forward(self, x, conv1w, conv1b, bn1w, bn1b, conv2w, conv2b, bn2w, bn2b, convdw=None, bndw=None, bndb=None):
        identity = x
        out = self.conv1._conv_forward(x, weight=conv1w, bias=conv1b)
        out = F.group_norm(out, self.bn1.num_groups, bn1w, bn1b, self.bn1.eps)
        out = self.relu(out)
        out = self.conv2._conv_forward(out, weight=conv2w, bias=conv2b)
        out = F.group_norm(out, self.bn2.num_groups, bn2w, bn2b, self.bn1.eps)
        if self.downsample is not None:
            identity = self.downsample[0]._conv_forward(x, weight=convdw, bias=None)
            identity = F.group_norm(identity, self.downsample[1].num_groups, bndw, bndb, self.downsample[1].eps)
        out += identity
        out = self.relu(out)
        return out

class CIFAR10Model:
    class AugmentDataset(Dataset):
        def __init__(self, dataset):
            self.dataset = dataset
            self.transform = torchvision.transforms.Compose(
                [RandomCrop(size=(32, 32), padding=4), RandomHorizontalFlip(0.5)])

        def __len__(self):
            return len(self.dataset)

        def __getitem__(self, item):
            img, label = self.dataset[item]
            return self.transform(img), label

    class ResNet(fuf.FModule):
        def __init__(
                self,
                block = PGBlock,
                layers = [2,2,2,2],
                zero_init_residual: bool = False,
                groups: int = 1,
                width_per_group: int = 64,
                replace_stride_with_dilation = None,
                norm_layer = lambda x: nn.GroupNorm(2, x),
        ) -> None:
            super().__init__()
            usub._log_api_usage_once(self)
            if norm_layer is None:
                norm_layer = nn.BatchNorm2d
            self._norm_layer = norm_layer

            self.inplanes = 64
            self.dilation = 1
            if replace_stride_with_dilation is None:
                # each element in the tuple indicates if we should replace
                # the 2x2 stride with a dilated convolution instead
                replace_stride_with_dilation = [False, False, False]
            if len(replace_stride_with_dilation) != 3:
                raise ValueError(
                    "replace_stride_with_dilation should be None "
                    f"or a 3-element tuple, got {replace_stride_with_dilation}"
                )
            self.groups = groups
            self.base_width = width_per_group
            self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
            self.bn1 = norm_layer(self.inplanes)
            self.relu = nn.ReLU(inplace=True)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            self.layer1 = self._make_layer(block, 64, layers[0])
            self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
            self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
            self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.head = nn.Linear(512 * block.expansion, 10)

            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

            # Zero-initialize the last BN in each residual branch,
            # so that the residual branch starts with zeros, and each residual block behaves like an identity.
            # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
            if zero_init_residual:
                for m in self.modules():
                    if isinstance(m, PGBlock) and m.bn2.weight is not None:
                        nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]
            self.adapted_model_para = {name: None for name, val in self.named_parameters()}

        def _make_layer(
                self,
                block,
                planes: int,
                blocks: int,
                stride: int = 1,
                dilate: bool = False,
        ) -> nn.Sequential:
            norm_layer = self._norm_layer
            downsample = None
            previous_dilation = self.dilation
            if dilate:
                self.dilation *= stride
                stride = 1
            if stride != 1 or self.inplanes != planes * block.expansion:
                downsample = nn.Sequential(
                    rn.conv1x1(self.inplanes, planes * block.expansion, stride),
                    norm_layer(planes * block.expansion),
                )

            layers = []
            layers.append(
                block(
                    self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation,
                    norm_layer
                )
            )
            self.inplanes = planes * block.expansion
            for _ in range(1, blocks):
                layers.append(
                    block(
                        self.inplanes,
                        planes,
                        groups=self.groups,
                        base_width=self.base_width,
                        dilation=self.dilation,
                        norm_layer=norm_layer,
                    )
                )

            return nn.Sequential(*layers)

        def adapted_forward(self, x):
            # See note [TorchScript super()]
            x = self.conv1._conv_forward(x, weight=self.adapted_model_para["conv1.weight"], bias=self.adapted_model_para.get("conv1.bias", None))
            x = F.group_norm(x, self.bn1.num_groups, self.adapted_model_para["bn1.weight"], self.adapted_model_para["bn1.bias"], self.bn1.eps)
            x = self.relu(x)
            x = self.maxpool(x)

            for i in range(4):
                layer = getattr(self, f'layer{i + 1}')
                for lid, li in enumerate(layer):
                    x = li.adapted_forward(
                        x,
                        self.adapted_model_para[f"layer{i+1}.{lid}.conv1.weight"],
                        self.adapted_model_para.get(f"layer{i+1}.{lid}.conv1.bias", None),
                        self.adapted_model_para[f"layer{i + 1}.{lid}.bn1.weight"],
                        self.adapted_model_para[f"layer{i + 1}.{lid}.bn1.bias"],
                        self.adapted_model_para[f"layer{i + 1}.{lid}.conv2.weight"],
                        self.adapted_model_para.get(f"layer{i + 1}.{lid}.conv2.bias", None),
                        self.adapted_model_para[f"layer{i + 1}.{lid}.bn2.weight"],
                        self.adapted_model_para[f"layer{i + 1}.{lid}.bn2.bias"],
                        self.adapted_model_para.get(f"layer{i + 1}.{lid}.downsample.0.weight", None),
                        self.adapted_model_para.get(f"layer{i + 1}.{lid}.downsample.1.weight", None),
                        self.adapted_model_para.get(f"layer{i + 1}.{lid}.downsample.1.bias", None),
                    )

            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = F.linear(x, weight=self.adapted_model_para["head.weight"], bias=self.adapted_model_para["head.bias"])

            return x

        def forward(self, x):
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)

            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)

            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)
            return x

        def set_adapted_para(self, name, val):
            self.adapted_model_para[name] = val

        def del_adapted_para(self):
            for key, val in self.adapted_model_para.items():
                if self.adapted_model_para[key] is not None:
                    self.adapted_model_para[key].grad = None
                    self.adapted_model_para[key] = None

    class Model(fuf.FModule):
        def __init__(self, sparse_factor=1.0, fine_grained_block_split=5):
            super().__init__()
            # basic modules
            self.model = CIFAR10Model.ResNet()
            self.gating_layer = GatingLayer(self.model, dataset_name='cifar10', fine_grained_block_split=fine_grained_block_split)
            self.total_model_size = torch.sum(self.gating_layer.block_size_lookup_table)
            # sparse factor
            self.sparse_factor = sparse_factor
            self.min_sparse_factor = min(max(1 / self.gating_layer.fine_grained_block_split, MIN_SPARSE_FACTOR), self.sparse_factor)
            # knapsack_solvel
            self.gated_scores_scale_factor = 10
            self.block_wise_prune = True
            self.knapsack_solver = KnapsackSolver01(
                value_sum_max=self.gated_scores_scale_factor * len(self.gating_layer.block_size_lookup_table),
                item_num_max=len(self.gating_layer.block_size_lookup_table),
                weight_max=round(self.sparse_factor * self.total_model_size.item())
            )

            # other attrs
            self.model_dim = int(self.get_param_tensor().shape[0])
            self.importance_prior_para_num = 0
            self.client_level_top_gated_scores = None
            self.gumbel_sigmoid = False
            self.person_input_norm = 0
            if self.person_input_norm == 1:
                self.gating_layer.norm_input_each_forward = False
            self.feed_batch_count = 0
            self.node_id = -1
            self.global_epoch = 0

        def get_param_tensor(self):
            param_list = []
            for param in self.model.parameters():
                param_list.append(param.data.view(-1, ))
            return torch.cat(param_list)

        def get_grad_tensor(self):
            grad_list = []

            for param in self.model.parameters():
                if param.grad is not None:
                    grad_list.append(param.grad.data.view(-1, ))
            return torch.cat(grad_list)

        def del_adapted_model_para(self):
            self.model.del_adapted_para()

        def get_top_gated_scores(self, x):
            """ Get gating weights via the learned gating layer data-dependently """
            # get gating weights data-dependently via gumbel trick
            gating_logits, trans_weights = self.gating_layer(x)  # -> [Batch_size, Num_blocks]
            if self.gumbel_sigmoid:
                # gumbel-sigmoid as softmax of two logits a and 0:  e^a / (e^a + e^0) = 1 / (1 + e^(0 - a)) = sigmoid(a)
                ori_logits_shape = gating_logits.size()
                gating_logits = torch.stack([torch.zeros(ori_logits_shape, device=gating_logits.device),
                                             gating_logits], dim=2)  # -> [Batch_size, Num_blocks, 2]
                gated_scores = gumbel_softmax(gating_logits, hard=False, dim=2)
                gated_scores = gated_scores * torch.stack(
                    [torch.zeros(ori_logits_shape, device=gating_logits.device),
                     torch.ones(ori_logits_shape, device=gating_logits.device)], dim=2)
                gated_scores = torch.sum(gated_scores, dim=2)  # -> [Batch_size, Num_blocks]
            else:
                # normed importance score
                gated_scores = torch.sigmoid(gating_logits)
            gated_scores = torch.mean(gated_scores, dim=0)  # -> [Num_blocks]

            # separate trans
            if id(gated_scores) != id(trans_weights):
                # bounded model diff
                trans_weights = torch.sigmoid(trans_weights)
                trans_weights = torch.mean(trans_weights, dim=0)  # -> [Num_blocks]

            # avoid cutting info flow (some internal sub-blocks are all zeros)
            gated_scores = torch.clip(gated_scores, min=self.min_sparse_factor)  # -> [Num_blocks]

            top_trans_weights, sparse_ratio_selected = self.select_top_trans_weights(gated_scores, trans_weights)

            return gated_scores, top_trans_weights, sparse_ratio_selected

        def adapt_prune_model(self, top_trans_weights):
            device = top_trans_weights.device
                # get pruned models via with ranked block-wise gating weights
            if self.gating_layer.fine_grained_block_split == 1:
                for para_idx, para in enumerate(self.model.parameters()):
                    mask = torch.ones_like(para, device=device).reshape(-1) * top_trans_weights[para_idx]
                    para_name = self.gating_layer.block_names[para_idx]
                    # self.model.adapted_model_para[para_name] = mask * para
                    mask = mask.view(para.shape)
                    self.model.set_adapted_para(para_name, mask * para)
            else:
                for para_name, para in self.model.named_parameters():
                    mask = torch.ones_like(para, device=device).reshape(-1)
                    sub_block_begin, sub_block_end, size_each_sub = self.gating_layer.para_name_to_block_split_info[
                        para_name]
                    for i in range(sub_block_begin, sub_block_end):
                        gating_weight_sub_block_i = top_trans_weights[i]
                        block_element_begin = (i - sub_block_begin) * size_each_sub
                        block_element_end = (i + 1 - sub_block_begin) * size_each_sub
                        mask[block_element_begin:block_element_end] *= gating_weight_sub_block_i
                    mask = mask.view(para.shape)
                    # self.model.adapted_model_para[para_name] = mask * para
                    self.model.set_adapted_para(para_name, mask * para)

            return top_trans_weights.detach()

        def select_top_trans_weights(self, gated_scores, trans_weight, in_place=True):
            """
            Keep to sefl.sparse_factor elements of gating weights
            :param gated_scores:
            :param in_place:
            :return:
            """
            device = gated_scores.device
            if self.sparse_factor == 1:
                return trans_weight, torch.tensor(1.0)
            if in_place:
                retained_trans_weights = trans_weight
            else:
                retained_trans_weights = trans_weight.clone()


            # keep top (self.sparse_factor) weights via 0-1 knapsack
            mask = torch.ones_like(gated_scores, device=device)
            if id(trans_weight) != id(gated_scores):
                # ST trick
                mask = mask - gated_scores.detach() + gated_scores
            if self.importance_prior_para_num == 1:
                importance_value_list = np.array(
                    ((gated_scores + self.gating_layer.block_size_lookup_table_normalized) / 2).tolist())
            else:
                importance_value_list = np.array(gated_scores.tolist())
            importance_value_list = np.around(importance_value_list * self.gated_scores_scale_factor).astype(int)

            # for linear_layer sub_blocks
            linear_layer_block_idx_filter_first = self.gating_layer.linear_layer_block_idx_filter_first
            selected_size = self._select_top_sub_blocks(importance_value_list, linear_layer_block_idx_filter_first,
                                                        mask)

            # for non-linear-layer sub_blocks
            non_linear_layer_block_idx_filter_first = self.gating_layer.non_linear_layer_block_idx_filter_first
            selected_size += self._select_top_sub_blocks(importance_value_list,
                                                         non_linear_layer_block_idx_filter_first,
                                                         mask)

            retained_trans_weights *= mask

            return retained_trans_weights, selected_size / self.total_model_size

        def _select_top_sub_blocks(self, importance_value_list, block_idx, mask):
            weight_list = self.gating_layer.block_size_lookup_table[block_idx]
            importance_value_list = importance_value_list[block_idx]
            capacity = torch.round(torch.sum(weight_list) * (self.sparse_factor - self.min_sparse_factor)).int()
            total_value_of_selected_items, total_weight, selected_item_idx, droped_item_idx = self.knapsack_solver.found_max_value_greedy(
                weight_list=weight_list.tolist(),
                value_list=importance_value_list,
                capacity=capacity
            )
            # droped_item_idx = [i for i in range(len(block_idx)) if
            #                    i not in selected_item_idx[0].tolist()]
            droped_item_idx = np.array(block_idx)[droped_item_idx]
            mask[droped_item_idx] *= 0

            if isinstance(total_weight, torch.Tensor):
                # return sum(weight_list[selected_item_idx]).detach()
                return total_weight.detach()
            else:
                return total_weight

        def _select_top_sub_blocks_frac(self, importance_value_list, block_idx, gated_scores_after_select):
            # to make the minimal gating weights of each block as self.min_sparse_factor,
            # we allocate the remaining capacity (self.sparse_model_size - self.min_sparse_factor)
            # into blocks according to their importance value (gating weights)
            weight_list = self.gating_layer.block_size_lookup_table[block_idx]
            importance_value_list = importance_value_list[block_idx]
            capacity = torch.sum(weight_list) * (self.sparse_factor - self.min_sparse_factor)
            total_value_of_selected_items, selected_items_weight, selected_items_frac = self.knapsack_solver.found_max_value(
                weight_list=weight_list * (1 - self.min_sparse_factor),
                value_list=importance_value_list,
                capacity=capacity
            )
            # to make the backward work, we add a calibration tensor onto gating weights,
            # such that the gated_scores close to the results from knapsack_solver
            gated_scores_after_select[block_idx] += selected_items_weight / weight_list

            return sum(selected_items_weight).detach()

        def forward(self, x):
            if self.person_input_norm:
                x = self.gating_layer.norm_input_layer(x)
            gated_scores, top_trans_weights, sparse_ratio_selected = self.get_top_gated_scores(x)
            # mask the meta-model according to sparsity preference
            self.adapt_prune_model(top_trans_weights)
            return self.model.adapted_forward(x)

    @classmethod
    def init_dataset(cls, object):
        if 'Client' in object.get_classname():
            object.train_data = cls.AugmentDataset(object.train_data)

    @classmethod
    def init_local_module(cls, object):
        if 'Client' in object.__class__.__name__:
            if not hasattr(object, '_model_class'):
                object._model_class = cls
                return
            else:
                return cls.Model(object._capacity, object.fine_grained_block_split)

    @classmethod
    def init_global_module(cls, object):
        if 'Server' in object.__class__.__name__:
            if not hasattr(object, '_model_class'):
                object._model_class = cls
                return
            else:
                return cls.Model(1.0, object.fine_grained_block_split)

class CIFAR100Model:
    class AugmentDataset(Dataset):
        def __init__(self, dataset):
            self.dataset = dataset
            self.transform = torchvision.transforms.Compose(
                [RandomCrop(size=(32, 32), padding=4), RandomHorizontalFlip(0.5)])

        def __len__(self):
            return len(self.dataset)

        def __getitem__(self, item):
            img, label = self.dataset[item]
            return self.transform(img), label

    class CNN(fuf.FModule):
        def __init__(self):
            super().__init__()
            self.maxpool = nn.MaxPool2d(2)
            self.relu = nn.ReLU()
            self.conv1 = nn.Conv2d(3, 64, 5)
            self.conv2 = nn.Conv2d(64, 64, 5)
            self.flatten = nn.Flatten(1)
            self.fc1 = nn.Linear(1600, 384)
            self.fc2 = nn.Linear(384, 192)
            self.head = nn.Linear(192, 100)
            self.adapted_model_para = {name: None for name, val in self.named_parameters()}

        def encoder(self, x):
            x = self.maxpool(self.relu(self.conv1(x)))
            x = self.maxpool(self.relu(self.conv2(x)))
            x = self.flatten(x)
            x = self.relu(self.fc1(x))
            x = self.relu(self.fc2(x))
            return x

        def forward(self, x):
            x = self.encoder(x)
            return self.head(x)

        def adapted_forward(self, x):
            # forward using the adapted parameters
            x = self.maxpool(F.relu(self.conv1._conv_forward(
                x, weight=self.adapted_model_para["conv1.weight"], bias=self.adapted_model_para["conv1.bias"])))
            x = self.maxpool(F.relu(self.conv2._conv_forward(
                x, weight=self.adapted_model_para["conv2.weight"], bias=self.adapted_model_para["conv2.bias"])))
            x = self.flatten(x)
            x = F.relu(F.linear(
                x, weight=self.adapted_model_para["fc1.weight"], bias=self.adapted_model_para["fc1.bias"]))
            x = F.relu(F.linear(
                x, weight=self.adapted_model_para["fc2.weight"], bias=self.adapted_model_para["fc2.bias"]))
            x = F.linear(
                x, weight=self.adapted_model_para["head.weight"], bias=self.adapted_model_para["head.bias"])
            return x

        def set_adapted_para(self, name, val):
            self.adapted_model_para[name] = val

        def del_adapted_para(self):
            for key, val in self.adapted_model_para.items():
                if self.adapted_model_para[key] is not None:
                    self.adapted_model_para[key].grad = None
                    self.adapted_model_para[key] = None

    class Model(fuf.FModule):
        def __init__(self, sparse_factor=1.0, fine_grained_block_split=100):
            super().__init__()
            # basic modules
            self.model = CIFAR100Model.CNN()
            self.gating_layer = GatingLayer(self.model, dataset_name='cifar100', fine_grained_block_split=fine_grained_block_split)
            self.total_model_size = torch.sum(self.gating_layer.block_size_lookup_table)
            # sparse factor
            self.sparse_factor = sparse_factor
            self.min_sparse_factor = min(max(1 / self.gating_layer.fine_grained_block_split, MIN_SPARSE_FACTOR), self.sparse_factor)
            # knapsack_solvel
            self.gated_scores_scale_factor = 10
            self.block_wise_prune = True
            self.knapsack_solver = KnapsackSolver01(
                value_sum_max=self.gated_scores_scale_factor * len(self.gating_layer.block_size_lookup_table),
                item_num_max=len(self.gating_layer.block_size_lookup_table),
                weight_max=round(self.sparse_factor * self.total_model_size.item())
            )

            # other attrs
            self.model_dim = int(self.get_param_tensor().shape[0])
            self.importance_prior_para_num = 0
            self.client_level_top_gated_scores = None
            self.gumbel_sigmoid = False
            self.person_input_norm = 0
            if self.person_input_norm == 1:
                self.gating_layer.norm_input_each_forward = False
            self.feed_batch_count = 0
            self.node_id = -1
            self.global_epoch = 0

        def get_param_tensor(self):
            param_list = []
            for param in self.model.parameters():
                param_list.append(param.data.view(-1, ))
            return torch.cat(param_list)

        def get_grad_tensor(self):
            grad_list = []

            for param in self.model.parameters():
                if param.grad is not None:
                    grad_list.append(param.grad.data.view(-1, ))
            return torch.cat(grad_list)

        def del_adapted_model_para(self):
            self.model.del_adapted_para()

        def get_top_gated_scores(self, x):
            """ Get gating weights via the learned gating layer data-dependently """
            # get gating weights data-dependently via gumbel trick
            gating_logits, trans_weights = self.gating_layer(x)  # -> [Batch_size, Num_blocks]
            if self.gumbel_sigmoid:
                # gumbel-sigmoid as softmax of two logits a and 0:  e^a / (e^a + e^0) = 1 / (1 + e^(0 - a)) = sigmoid(a)
                ori_logits_shape = gating_logits.size()
                gating_logits = torch.stack([torch.zeros(ori_logits_shape, device=gating_logits.device),
                                             gating_logits], dim=2)  # -> [Batch_size, Num_blocks, 2]
                gated_scores = gumbel_softmax(gating_logits, hard=False, dim=2)
                gated_scores = gated_scores * torch.stack(
                    [torch.zeros(ori_logits_shape, device=gating_logits.device),
                     torch.ones(ori_logits_shape, device=gating_logits.device)], dim=2)
                gated_scores = torch.sum(gated_scores, dim=2)  # -> [Batch_size, Num_blocks]
            else:
                # normed importance score
                gated_scores = torch.sigmoid(gating_logits)
            gated_scores = torch.mean(gated_scores, dim=0)  # -> [Num_blocks]

            # separate trans
            if id(gated_scores) != id(trans_weights):
                # bounded model diff
                trans_weights = torch.sigmoid(trans_weights)
                trans_weights = torch.mean(trans_weights, dim=0)  # -> [Num_blocks]

            # avoid cutting info flow (some internal sub-blocks are all zeros)
            gated_scores = torch.clip(gated_scores, min=self.min_sparse_factor)  # -> [Num_blocks]

            top_trans_weights, sparse_ratio_selected = self.select_top_trans_weights(gated_scores, trans_weights)

            return gated_scores, top_trans_weights, sparse_ratio_selected

        def adapt_prune_model(self, top_trans_weights):
            device = top_trans_weights.device
                # get pruned models via with ranked block-wise gating weights
            if self.gating_layer.fine_grained_block_split == 1:
                for para_idx, para in enumerate(self.model.parameters()):
                    mask = torch.ones_like(para, device=device).reshape(-1) * top_trans_weights[para_idx]
                    para_name = self.gating_layer.block_names[para_idx]
                    # self.model.adapted_model_para[para_name] = mask * para
                    mask = mask.view(para.shape)
                    self.model.set_adapted_para(para_name, mask * para)
            else:
                for para_name, para in self.model.named_parameters():
                    mask = torch.ones_like(para, device=device).reshape(-1)
                    sub_block_begin, sub_block_end, size_each_sub = self.gating_layer.para_name_to_block_split_info[
                        para_name]
                    for i in range(sub_block_begin, sub_block_end):
                        gating_weight_sub_block_i = top_trans_weights[i]
                        block_element_begin = (i - sub_block_begin) * size_each_sub
                        block_element_end = (i + 1 - sub_block_begin) * size_each_sub
                        mask[block_element_begin:block_element_end] *= gating_weight_sub_block_i
                    mask = mask.view(para.shape)
                    # self.model.adapted_model_para[para_name] = mask * para
                    self.model.set_adapted_para(para_name, mask * para)

            return top_trans_weights.detach()

        def select_top_trans_weights(self, gated_scores, trans_weight, in_place=True):
            """
            Keep to sefl.sparse_factor elements of gating weights
            :param gated_scores:
            :param in_place:
            :return:
            """
            device = gated_scores.device
            if self.sparse_factor == 1:
                return trans_weight, torch.tensor(1.0)
            if in_place:
                retained_trans_weights = trans_weight
            else:
                retained_trans_weights = trans_weight.clone()


            # keep top (self.sparse_factor) weights via 0-1 knapsack
            mask = torch.ones_like(gated_scores, device=device)
            if id(trans_weight) != id(gated_scores):
                # ST trick
                mask = mask - gated_scores.detach() + gated_scores
            if self.importance_prior_para_num == 1:
                importance_value_list = np.array(
                    ((gated_scores + self.gating_layer.block_size_lookup_table_normalized) / 2).tolist())
            else:
                importance_value_list = np.array(gated_scores.tolist())
            importance_value_list = np.around(importance_value_list * self.gated_scores_scale_factor).astype(int)

            # for linear_layer sub_blocks
            linear_layer_block_idx_filter_first = self.gating_layer.linear_layer_block_idx_filter_first
            selected_size = self._select_top_sub_blocks(importance_value_list, linear_layer_block_idx_filter_first,
                                                        mask)

            # for non-linear-layer sub_blocks
            non_linear_layer_block_idx_filter_first = self.gating_layer.non_linear_layer_block_idx_filter_first
            selected_size += self._select_top_sub_blocks(importance_value_list,
                                                         non_linear_layer_block_idx_filter_first,
                                                         mask)

            retained_trans_weights *= mask

            return retained_trans_weights, selected_size / self.total_model_size

        def _select_top_sub_blocks(self, importance_value_list, block_idx, mask):
            weight_list = self.gating_layer.block_size_lookup_table[block_idx]
            importance_value_list = importance_value_list[block_idx]
            capacity = torch.round(torch.sum(weight_list) * (self.sparse_factor - self.min_sparse_factor)).int()
            total_value_of_selected_items, total_weight, selected_item_idx, droped_item_idx = self.knapsack_solver.found_max_value_greedy(
                weight_list=weight_list.tolist(),
                value_list=importance_value_list,
                capacity=capacity
            )
            # droped_item_idx = [i for i in range(len(block_idx)) if
            #                    i not in selected_item_idx[0].tolist()]
            droped_item_idx = np.array(block_idx)[droped_item_idx]
            mask[droped_item_idx] *= 0

            if isinstance(total_weight, torch.Tensor):
                # return sum(weight_list[selected_item_idx]).detach()
                return total_weight.detach()
            else:
                return total_weight

        def _select_top_sub_blocks_frac(self, importance_value_list, block_idx, gated_scores_after_select):
            # to make the minimal gating weights of each block as self.min_sparse_factor,
            # we allocate the remaining capacity (self.sparse_model_size - self.min_sparse_factor)
            # into blocks according to their importance value (gating weights)
            weight_list = self.gating_layer.block_size_lookup_table[block_idx]
            importance_value_list = importance_value_list[block_idx]
            capacity = torch.sum(weight_list) * (self.sparse_factor - self.min_sparse_factor)
            total_value_of_selected_items, selected_items_weight, selected_items_frac = self.knapsack_solver.found_max_value(
                weight_list=weight_list * (1 - self.min_sparse_factor),
                value_list=importance_value_list,
                capacity=capacity
            )
            # to make the backward work, we add a calibration tensor onto gating weights,
            # such that the gated_scores close to the results from knapsack_solver
            gated_scores_after_select[block_idx] += selected_items_weight / weight_list

            return sum(selected_items_weight).detach()

        def forward(self, x):
            if self.person_input_norm:
                x = self.gating_layer.norm_input_layer(x)
            gated_scores, top_trans_weights, sparse_ratio_selected = self.get_top_gated_scores(x)
            # mask the meta-model according to sparsity preference
            self.adapt_prune_model(top_trans_weights)
            return self.model.adapted_forward(x)

    @classmethod
    def init_dataset(cls, object):
        if 'Client' in object.get_classname():
            object.train_data = cls.AugmentDataset(object.train_data)

    @classmethod
    def init_local_module(cls, object):
        if 'Client' in object.__class__.__name__:
            if not hasattr(object, '_model_class'):
                object._model_class = cls
                return
            else:
                return cls.Model(object._capacity, object.fine_grained_block_split)

    @classmethod
    def init_global_module(cls, object):
        if 'Server' in object.__class__.__name__:
            if not hasattr(object, '_model_class'):
                object._model_class = cls
                return
            else:
                return cls.Model(1.0, object.fine_grained_block_split)

class DOMAINNETModel:
    class AlexNet(fuf.FModule):
        """
        used for DomainNet and Office-Caltech10
        """
        def __init__(self, num_classes=10):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2)
            self.conv2 = nn.Conv2d(64, 192, kernel_size=5, padding=2)
            self.conv3 = nn.Conv2d(192, 384, kernel_size=3, padding=1)
            self.conv4 = nn.Conv2d(384, 256, kernel_size=3, padding=1)
            self.conv5 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
            self.relu = nn.ReLU()
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2)
            self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
            self.fc1 = nn.Linear(256 * 6 * 6, 1024)
            self.fc2 = nn.Linear(1024, 1024)
            self.fc3 = nn.Linear(1024, num_classes)

        def forward(self, x):
            x = self.conv1(x)
            x = self.relu(x)
            x = self.maxpool(x)
            x = self.conv2(x)
            x = self.relu(x)
            x = self.maxpool(x)
            x = self.conv3(x)
            x = self.relu(x)
            x = self.conv4(x)
            x = self.relu(x)
            x = self.conv5(x)
            x = self.relu(x)
            x = self.maxpool(x)
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc1(x)
            x = self.relu(x)
            x = self.fc2(x)
            x = self.relu(x)
            x = self.fc3(x)
            return x

        def adapted_forward(self, x):
            # forward using the adapted parameters
            x = self.maxpool(F.relu(self.conv1._conv_forward(
                x, weight=self.adapted_model_para["conv1.weight"], bias=self.adapted_model_para["conv1.bias"])))
            x = self.maxpool(F.relu(self.conv2._conv_forward(
                x, weight=self.adapted_model_para["conv2.weight"], bias=self.adapted_model_para["conv2.bias"])))
            x = F.relu(self.conv3._conv_forward(
                x, weight=self.adapted_model_para["conv3.weight"], bias=self.adapted_model_para["conv3.bias"]))
            x = F.relu(self.conv4._conv_forward(
                x, weight=self.adapted_model_para["conv4.weight"], bias=self.adapted_model_para["conv4.bias"]))
            x = self.maxpool(F.relu(self.conv5._conv_forward(
                x, weight=self.adapted_model_para["conv5.weight"], bias=self.adapted_model_para["conv5.bias"])))
            x = self.flatten(x)
            x = F.relu(F.linear(
                x, weight=self.adapted_model_para["fc1.weight"], bias=self.adapted_model_para["fc1.bias"]))
            x = F.relu(F.linear(
                x, weight=self.adapted_model_para["fc2.weight"], bias=self.adapted_model_para["fc2.bias"]))
            x = F.linear(
                x, weight=self.adapted_model_para["head.weight"], bias=self.adapted_model_para["head.bias"])
            return x

        def set_adapted_para(self, name, val):
            self.adapted_model_para[name] = val

        def del_adapted_para(self):
            for key, val in self.adapted_model_para.items():
                if self.adapted_model_para[key] is not None:
                    self.adapted_model_para[key].grad = None
                    self.adapted_model_para[key] = None

    @classmethod
    def init_dataset(cls, object):
        pass

    @classmethod
    def init_local_module(cls, object):
        pass

    @classmethod
    def init_global_module(cls, object):
        if 'Server' in object.__class__.__name__:
            object.model = cls.AlexNet().to(object.device)

class MNISTModel:
    class CNN(fuf.FModule):
        def __init__(self):
            super().__init__()
            self.maxpool = nn.MaxPool2d(2)
            self.relu = nn.ReLU()
            self.conv1 = nn.Conv2d(1, 32, 5, padding=2)
            self.conv2 = nn.Conv2d(32, 64, 5, padding=2)
            self.flatten = nn.Flatten(1)
            self.fc1 = nn.Linear(3136, 512)
            self.fc2 = nn.Linear(512, 128)
            self.head = nn.Linear(128, 10)
            self.adapted_model_para = {name: None for name, val in self.named_parameters()}

        def encoder(self, x):
            x = self.maxpool(self.relu(self.conv1(x)))
            x = self.maxpool(self.relu(self.conv2(x)))
            x = self.flatten(x)
            x = self.relu(self.fc1(x))
            x = self.relu(self.fc2(x))
            return x

        def forward(self, x):
            x = self.encoder(x)
            return self.head(x)

        def adapted_forward(self, x):
            # forward using the adapted parameters
            x = self.maxpool(F.relu(self.conv1._conv_forward(
                x, weight=self.adapted_model_para["conv1.weight"], bias=self.adapted_model_para["conv1.bias"])))
            x = self.maxpool(F.relu(self.conv2._conv_forward(
                x, weight=self.adapted_model_para["conv2.weight"], bias=self.adapted_model_para["conv2.bias"])))
            x = self.flatten(x)
            x = F.relu(F.linear(
                x, weight=self.adapted_model_para["fc1.weight"], bias=self.adapted_model_para["fc1.bias"]))
            x = F.relu(F.linear(
                x, weight=self.adapted_model_para["fc2.weight"], bias=self.adapted_model_para["fc2.bias"]))
            x = F.linear(
                x, weight=self.adapted_model_para["head.weight"], bias=self.adapted_model_para["head.bias"])
            return x

        def set_adapted_para(self, name, val):
            self.adapted_model_para[name] = val

        def del_adapted_para(self):
            for key, val in self.adapted_model_para.items():
                if self.adapted_model_para[key] is not None:
                    self.adapted_model_para[key].grad = None
                    self.adapted_model_para[key] = None

    class Model(fuf.FModule):
        def __init__(self, sparse_factor=1.0, fine_grained_block_split=100):
            super().__init__()
            # basic modules
            self.model = MNISTModel.CNN()
            self.gating_layer = GatingLayer(self.model, dataset_name='mnist', fine_grained_block_split=fine_grained_block_split)
            self.total_model_size = torch.sum(self.gating_layer.block_size_lookup_table)
            # sparse factor
            self.sparse_factor = sparse_factor
            self.min_sparse_factor = min(max(1 / self.gating_layer.fine_grained_block_split, MIN_SPARSE_FACTOR), self.sparse_factor)
            # knapsack_solvel
            self.gated_scores_scale_factor = 10
            self.block_wise_prune = True
            self.knapsack_solver = KnapsackSolver01(
                value_sum_max=self.gated_scores_scale_factor * len(self.gating_layer.block_size_lookup_table),
                item_num_max=len(self.gating_layer.block_size_lookup_table),
                weight_max=round(self.sparse_factor * self.total_model_size.item())
            )

            # other attrs
            self.model_dim = int(self.get_param_tensor().shape[0])
            self.importance_prior_para_num = 0
            self.client_level_top_gated_scores = None
            self.gumbel_sigmoid = False
            self.person_input_norm = 0
            if self.person_input_norm == 1:
                self.gating_layer.norm_input_each_forward = False
            self.feed_batch_count = 0
            self.node_id = -1
            self.global_epoch = 0

        def get_param_tensor(self):
            param_list = []
            for param in self.model.parameters():
                param_list.append(param.data.view(-1, ))
            return torch.cat(param_list)

        def get_grad_tensor(self):
            grad_list = []

            for param in self.model.parameters():
                if param.grad is not None:
                    grad_list.append(param.grad.data.view(-1, ))
            return torch.cat(grad_list)

        def del_adapted_model_para(self):
            self.model.del_adapted_para()

        def get_top_gated_scores(self, x):
            """ Get gating weights via the learned gating layer data-dependently """
            # get gating weights data-dependently via gumbel trick
            gating_logits, trans_weights = self.gating_layer(x)  # -> [Batch_size, Num_blocks]
            if self.gumbel_sigmoid:
                # gumbel-sigmoid as softmax of two logits a and 0:  e^a / (e^a + e^0) = 1 / (1 + e^(0 - a)) = sigmoid(a)
                ori_logits_shape = gating_logits.size()
                gating_logits = torch.stack([torch.zeros(ori_logits_shape, device=gating_logits.device),
                                             gating_logits], dim=2)  # -> [Batch_size, Num_blocks, 2]
                gated_scores = gumbel_softmax(gating_logits, hard=False, dim=2)
                gated_scores = gated_scores * torch.stack(
                    [torch.zeros(ori_logits_shape, device=gating_logits.device),
                     torch.ones(ori_logits_shape, device=gating_logits.device)], dim=2)
                gated_scores = torch.sum(gated_scores, dim=2)  # -> [Batch_size, Num_blocks]
            else:
                # normed importance score
                gated_scores = torch.sigmoid(gating_logits)
            gated_scores = torch.mean(gated_scores, dim=0)  # -> [Num_blocks]

            # separate trans
            if id(gated_scores) != id(trans_weights):
                # bounded model diff
                trans_weights = torch.sigmoid(trans_weights)
                trans_weights = torch.mean(trans_weights, dim=0)  # -> [Num_blocks]

            # avoid cutting info flow (some internal sub-blocks are all zeros)
            gated_scores = torch.clip(gated_scores, min=self.min_sparse_factor)  # -> [Num_blocks]

            top_trans_weights, sparse_ratio_selected = self.select_top_trans_weights(gated_scores, trans_weights)

            return gated_scores, top_trans_weights, sparse_ratio_selected

        def adapt_prune_model(self, top_trans_weights):
            device = top_trans_weights.device
                # get pruned models via with ranked block-wise gating weights
            if self.gating_layer.fine_grained_block_split == 1:
                for para_idx, para in enumerate(self.model.parameters()):
                    mask = torch.ones_like(para, device=device).reshape(-1) * top_trans_weights[para_idx]
                    para_name = self.gating_layer.block_names[para_idx]
                    # self.model.adapted_model_para[para_name] = mask * para
                    mask = mask.view(para.shape)
                    self.model.set_adapted_para(para_name, mask * para)
            else:
                for para_name, para in self.model.named_parameters():
                    mask = torch.ones_like(para, device=device).reshape(-1)
                    sub_block_begin, sub_block_end, size_each_sub = self.gating_layer.para_name_to_block_split_info[
                        para_name]
                    for i in range(sub_block_begin, sub_block_end):
                        gating_weight_sub_block_i = top_trans_weights[i]
                        block_element_begin = (i - sub_block_begin) * size_each_sub
                        block_element_end = (i + 1 - sub_block_begin) * size_each_sub
                        mask[block_element_begin:block_element_end] *= gating_weight_sub_block_i
                    mask = mask.view(para.shape)
                    # self.model.adapted_model_para[para_name] = mask * para
                    self.model.set_adapted_para(para_name, mask * para)

            return top_trans_weights.detach()

        def select_top_trans_weights(self, gated_scores, trans_weight, in_place=True):
            """
            Keep to sefl.sparse_factor elements of gating weights
            :param gated_scores:
            :param in_place:
            :return:
            """
            device = gated_scores.device
            if self.sparse_factor == 1:
                return trans_weight, torch.tensor(1.0)
            if in_place:
                retained_trans_weights = trans_weight
            else:
                retained_trans_weights = trans_weight.clone()


            # keep top (self.sparse_factor) weights via 0-1 knapsack
            mask = torch.ones_like(gated_scores, device=device)
            if id(trans_weight) != id(gated_scores):
                # ST trick
                mask = mask - gated_scores.detach() + gated_scores
            if self.importance_prior_para_num == 1:
                importance_value_list = np.array(
                    ((gated_scores + self.gating_layer.block_size_lookup_table_normalized) / 2).tolist())
            else:
                importance_value_list = np.array(gated_scores.tolist())
            importance_value_list = np.around(importance_value_list * self.gated_scores_scale_factor).astype(int)

            # for linear_layer sub_blocks
            linear_layer_block_idx_filter_first = self.gating_layer.linear_layer_block_idx_filter_first
            selected_size = self._select_top_sub_blocks(importance_value_list, linear_layer_block_idx_filter_first,
                                                        mask)

            # for non-linear-layer sub_blocks
            non_linear_layer_block_idx_filter_first = self.gating_layer.non_linear_layer_block_idx_filter_first
            selected_size += self._select_top_sub_blocks(importance_value_list,
                                                         non_linear_layer_block_idx_filter_first,
                                                         mask)

            retained_trans_weights *= mask

            return retained_trans_weights, selected_size / self.total_model_size

        def _select_top_sub_blocks(self, importance_value_list, block_idx, mask):
            weight_list = self.gating_layer.block_size_lookup_table[block_idx]
            importance_value_list = importance_value_list[block_idx]
            capacity = torch.round(torch.sum(weight_list) * (self.sparse_factor - self.min_sparse_factor)).int()
            total_value_of_selected_items, total_weight, selected_item_idx, droped_item_idx = self.knapsack_solver.found_max_value_greedy(
                weight_list=weight_list.tolist(),
                value_list=importance_value_list,
                capacity=capacity
            )
            # droped_item_idx = [i for i in range(len(block_idx)) if
            #                    i not in selected_item_idx[0].tolist()]
            droped_item_idx = np.array(block_idx)[droped_item_idx]
            mask[droped_item_idx] *= 0

            if isinstance(total_weight, torch.Tensor):
                # return sum(weight_list[selected_item_idx]).detach()
                return total_weight.detach()
            else:
                return total_weight

        def _select_top_sub_blocks_frac(self, importance_value_list, block_idx, gated_scores_after_select):
            # to make the minimal gating weights of each block as self.min_sparse_factor,
            # we allocate the remaining capacity (self.sparse_model_size - self.min_sparse_factor)
            # into blocks according to their importance value (gating weights)
            weight_list = self.gating_layer.block_size_lookup_table[block_idx]
            importance_value_list = importance_value_list[block_idx]
            capacity = torch.sum(weight_list) * (self.sparse_factor - self.min_sparse_factor)
            total_value_of_selected_items, selected_items_weight, selected_items_frac = self.knapsack_solver.found_max_value(
                weight_list=weight_list * (1 - self.min_sparse_factor),
                value_list=importance_value_list,
                capacity=capacity
            )
            # to make the backward work, we add a calibration tensor onto gating weights,
            # such that the gated_scores close to the results from knapsack_solver
            gated_scores_after_select[block_idx] += selected_items_weight / weight_list

            return sum(selected_items_weight).detach()

        def forward(self, x):
            if self.person_input_norm:
                x = self.gating_layer.norm_input_layer(x)
            gated_scores, top_trans_weights, sparse_ratio_selected = self.get_top_gated_scores(x)
            # mask the meta-model according to sparsity preference
            self.adapt_prune_model(top_trans_weights)
            return self.model.adapted_forward(x)

    @classmethod
    def init_dataset(cls, object):
        pass

    @classmethod
    def init_local_module(cls, object):
        if 'Client' in object.__class__.__name__:
            if not hasattr(object, '_model_class'):
                object._model_class = cls
                return
            else:
                return cls.Model(object._capacity, object.fine_grained_block_split)

    @classmethod
    def init_global_module(cls, object):
        if 'Server' in object.__class__.__name__:
            if not hasattr(object, '_model_class'):
                object._model_class = cls
                return
            else:
                return cls.Model(1.0, object.fine_grained_block_split)

class FASHIONModel:
    class CNN(fuf.FModule):
        def __init__(self):
            super().__init__()
            self.maxpool = nn.MaxPool2d(2)
            self.relu = nn.ReLU()
            self.conv1 = nn.Conv2d(1, 32, 5, padding=2)
            self.conv2 = nn.Conv2d(32, 64, 5, padding=2)
            self.flatten = nn.Flatten(1)
            self.fc1 = nn.Linear(3136, 512)
            self.fc2 = nn.Linear(512, 128)
            self.head = nn.Linear(128, 10)
            self.adapted_model_para = {name: None for name, val in self.named_parameters()}

        def encoder(self, x):
            x = self.maxpool(self.relu(self.conv1(x)))
            x = self.maxpool(self.relu(self.conv2(x)))
            x = self.flatten(x)
            x = self.relu(self.fc1(x))
            x = self.relu(self.fc2(x))
            return x

        def forward(self, x):
            x = self.encoder(x)
            return self.head(x)

        def adapted_forward(self, x):
            # forward using the adapted parameters
            x = self.maxpool(F.relu(self.conv1._conv_forward(
                x, weight=self.adapted_model_para["conv1.weight"], bias=self.adapted_model_para["conv1.bias"])))
            x = self.maxpool(F.relu(self.conv2._conv_forward(
                x, weight=self.adapted_model_para["conv2.weight"], bias=self.adapted_model_para["conv2.bias"])))
            x = self.flatten(x)
            x = F.relu(F.linear(
                x, weight=self.adapted_model_para["fc1.weight"], bias=self.adapted_model_para["fc1.bias"]))
            x = F.relu(F.linear(
                x, weight=self.adapted_model_para["fc2.weight"], bias=self.adapted_model_para["fc2.bias"]))
            x = F.linear(
                x, weight=self.adapted_model_para["head.weight"], bias=self.adapted_model_para["head.bias"])
            return x

        def set_adapted_para(self, name, val):
            self.adapted_model_para[name] = val

        def del_adapted_para(self):
            for key, val in self.adapted_model_para.items():
                if self.adapted_model_para[key] is not None:
                    self.adapted_model_para[key].grad = None
                    self.adapted_model_para[key] = None

    class Model(fuf.FModule):
        def __init__(self, sparse_factor=1.0, fine_grained_block_split=100):
            super().__init__()
            # basic modules
            self.model = FASHIONModel.CNN()
            self.gating_layer = GatingLayer(self.model, dataset_name='fashion', fine_grained_block_split=fine_grained_block_split)
            self.total_model_size = torch.sum(self.gating_layer.block_size_lookup_table)
            # sparse factor
            self.sparse_factor = sparse_factor
            self.min_sparse_factor = min(max(1 / self.gating_layer.fine_grained_block_split, MIN_SPARSE_FACTOR), self.sparse_factor)
            # knapsack_solvel
            self.gated_scores_scale_factor = 10
            self.block_wise_prune = True
            self.knapsack_solver = KnapsackSolver01(
                value_sum_max=self.gated_scores_scale_factor * len(self.gating_layer.block_size_lookup_table),
                item_num_max=len(self.gating_layer.block_size_lookup_table),
                weight_max=round(self.sparse_factor * self.total_model_size.item())
            )

            # other attrs
            self.model_dim = int(self.get_param_tensor().shape[0])
            self.importance_prior_para_num = 0
            self.client_level_top_gated_scores = None
            self.gumbel_sigmoid = False
            self.person_input_norm = 0
            if self.person_input_norm == 1:
                self.gating_layer.norm_input_each_forward = False
            self.feed_batch_count = 0
            self.node_id = -1
            self.global_epoch = 0

        def get_param_tensor(self):
            param_list = []
            for param in self.model.parameters():
                param_list.append(param.data.view(-1, ))
            return torch.cat(param_list)

        def get_grad_tensor(self):
            grad_list = []

            for param in self.model.parameters():
                if param.grad is not None:
                    grad_list.append(param.grad.data.view(-1, ))
            return torch.cat(grad_list)

        def del_adapted_model_para(self):
            self.model.del_adapted_para()

        def get_top_gated_scores(self, x):
            """ Get gating weights via the learned gating layer data-dependently """
            # get gating weights data-dependently via gumbel trick
            gating_logits, trans_weights = self.gating_layer(x)  # -> [Batch_size, Num_blocks]
            if self.gumbel_sigmoid:
                # gumbel-sigmoid as softmax of two logits a and 0:  e^a / (e^a + e^0) = 1 / (1 + e^(0 - a)) = sigmoid(a)
                ori_logits_shape = gating_logits.size()
                gating_logits = torch.stack([torch.zeros(ori_logits_shape, device=gating_logits.device),
                                             gating_logits], dim=2)  # -> [Batch_size, Num_blocks, 2]
                gated_scores = gumbel_softmax(gating_logits, hard=False, dim=2)
                gated_scores = gated_scores * torch.stack(
                    [torch.zeros(ori_logits_shape, device=gating_logits.device),
                     torch.ones(ori_logits_shape, device=gating_logits.device)], dim=2)
                gated_scores = torch.sum(gated_scores, dim=2)  # -> [Batch_size, Num_blocks]
            else:
                # normed importance score
                gated_scores = torch.sigmoid(gating_logits)
            gated_scores = torch.mean(gated_scores, dim=0)  # -> [Num_blocks]

            # separate trans
            if id(gated_scores) != id(trans_weights):
                # bounded model diff
                trans_weights = torch.sigmoid(trans_weights)
                trans_weights = torch.mean(trans_weights, dim=0)  # -> [Num_blocks]

            # avoid cutting info flow (some internal sub-blocks are all zeros)
            gated_scores = torch.clip(gated_scores, min=self.min_sparse_factor)  # -> [Num_blocks]

            top_trans_weights, sparse_ratio_selected = self.select_top_trans_weights(gated_scores, trans_weights)

            return gated_scores, top_trans_weights, sparse_ratio_selected

        def adapt_prune_model(self, top_trans_weights):
            device = top_trans_weights.device
                # get pruned models via with ranked block-wise gating weights
            if self.gating_layer.fine_grained_block_split == 1:
                for para_idx, para in enumerate(self.model.parameters()):
                    mask = torch.ones_like(para, device=device).reshape(-1) * top_trans_weights[para_idx]
                    para_name = self.gating_layer.block_names[para_idx]
                    # self.model.adapted_model_para[para_name] = mask * para
                    mask = mask.view(para.shape)
                    self.model.set_adapted_para(para_name, mask * para)
            else:
                for para_name, para in self.model.named_parameters():
                    mask = torch.ones_like(para, device=device).reshape(-1)
                    sub_block_begin, sub_block_end, size_each_sub = self.gating_layer.para_name_to_block_split_info[
                        para_name]
                    for i in range(sub_block_begin, sub_block_end):
                        gating_weight_sub_block_i = top_trans_weights[i]
                        block_element_begin = (i - sub_block_begin) * size_each_sub
                        block_element_end = (i + 1 - sub_block_begin) * size_each_sub
                        mask[block_element_begin:block_element_end] *= gating_weight_sub_block_i
                    mask = mask.view(para.shape)
                    # self.model.adapted_model_para[para_name] = mask * para
                    self.model.set_adapted_para(para_name, mask * para)

            return top_trans_weights.detach()

        def select_top_trans_weights(self, gated_scores, trans_weight, in_place=True):
            """
            Keep to sefl.sparse_factor elements of gating weights
            :param gated_scores:
            :param in_place:
            :return:
            """
            device = gated_scores.device
            if self.sparse_factor == 1:
                return trans_weight, torch.tensor(1.0)
            if in_place:
                retained_trans_weights = trans_weight
            else:
                retained_trans_weights = trans_weight.clone()


            # keep top (self.sparse_factor) weights via 0-1 knapsack
            mask = torch.ones_like(gated_scores, device=device)
            if id(trans_weight) != id(gated_scores):
                # ST trick
                mask = mask - gated_scores.detach() + gated_scores
            if self.importance_prior_para_num == 1:
                importance_value_list = np.array(
                    ((gated_scores + self.gating_layer.block_size_lookup_table_normalized) / 2).tolist())
            else:
                importance_value_list = np.array(gated_scores.tolist())
            importance_value_list = np.around(importance_value_list * self.gated_scores_scale_factor).astype(int)

            # for linear_layer sub_blocks
            linear_layer_block_idx_filter_first = self.gating_layer.linear_layer_block_idx_filter_first
            selected_size = self._select_top_sub_blocks(importance_value_list, linear_layer_block_idx_filter_first,
                                                        mask)

            # for non-linear-layer sub_blocks
            non_linear_layer_block_idx_filter_first = self.gating_layer.non_linear_layer_block_idx_filter_first
            selected_size += self._select_top_sub_blocks(importance_value_list,
                                                         non_linear_layer_block_idx_filter_first,
                                                         mask)

            retained_trans_weights *= mask

            return retained_trans_weights, selected_size / self.total_model_size

        def _select_top_sub_blocks(self, importance_value_list, block_idx, mask):
            weight_list = self.gating_layer.block_size_lookup_table[block_idx]
            importance_value_list = importance_value_list[block_idx]
            capacity = torch.round(torch.sum(weight_list) * (self.sparse_factor - self.min_sparse_factor)).int()
            total_value_of_selected_items, total_weight, selected_item_idx, droped_item_idx = self.knapsack_solver.found_max_value_greedy(
                weight_list=weight_list.tolist(),
                value_list=importance_value_list,
                capacity=capacity
            )
            # droped_item_idx = [i for i in range(len(block_idx)) if
            #                    i not in selected_item_idx[0].tolist()]
            droped_item_idx = np.array(block_idx)[droped_item_idx]
            mask[droped_item_idx] *= 0

            if isinstance(total_weight, torch.Tensor):
                # return sum(weight_list[selected_item_idx]).detach()
                return total_weight.detach()
            else:
                return total_weight

        def _select_top_sub_blocks_frac(self, importance_value_list, block_idx, gated_scores_after_select):
            # to make the minimal gating weights of each block as self.min_sparse_factor,
            # we allocate the remaining capacity (self.sparse_model_size - self.min_sparse_factor)
            # into blocks according to their importance value (gating weights)
            weight_list = self.gating_layer.block_size_lookup_table[block_idx]
            importance_value_list = importance_value_list[block_idx]
            capacity = torch.sum(weight_list) * (self.sparse_factor - self.min_sparse_factor)
            total_value_of_selected_items, selected_items_weight, selected_items_frac = self.knapsack_solver.found_max_value(
                weight_list=weight_list * (1 - self.min_sparse_factor),
                value_list=importance_value_list,
                capacity=capacity
            )
            # to make the backward work, we add a calibration tensor onto gating weights,
            # such that the gated_scores close to the results from knapsack_solver
            gated_scores_after_select[block_idx] += selected_items_weight / weight_list

            return sum(selected_items_weight).detach()

        def forward(self, x):
            if self.person_input_norm:
                x = self.gating_layer.norm_input_layer(x)
            gated_scores, top_trans_weights, sparse_ratio_selected = self.get_top_gated_scores(x)
            # mask the meta-model according to sparsity preference
            self.adapt_prune_model(top_trans_weights)
            return self.model.adapted_forward(x)

    @classmethod
    def init_dataset(cls, object):
        pass

    @classmethod
    def init_local_module(cls, object):
        if 'Client' in object.__class__.__name__:
            if not hasattr(object, '_model_class'):
                object._model_class = cls
                return
            else:
                return cls.Model(object._capacity, object.fine_grained_block_split)

    @classmethod
    def init_global_module(cls, object):
        if 'Server' in object.__class__.__name__:
            if not hasattr(object, '_model_class'):
                object._model_class = cls
                return
            else:
                return cls.Model(1.0, object.fine_grained_block_split)



