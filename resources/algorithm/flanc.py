import copy
import os
import torch
import copy
import torch.nn
import flgo.utils.fmodule as fuf
from torch.utils.data import Dataset
from torchvision.transforms import RandomCrop, RandomHorizontalFlip
import torchvision
import flgo.algorithm.fedavg as fedavg
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import flgo.utils.submodule as usub

class Server(fedavg.Server):
    def initialize(self, *args, **kwargs):
        self.init_algo_para({"lmbd": 1e-4, 'R2frac':0.25, 'minp':-1, 'R1frac_ratio':1.0})
        self.client_ps = []
        if self.minp<0:
            min_caps = min([c._capacity for c in self.clients])
            dp_list = [1.0/(2**i) for i in range(6)]
            i = 0
            while i<len(dp_list) and dp_list[i]**2>min_caps:
                i+=1
            if i==len(dp_list): raise RuntimeError(f"Failed to train with Client with lowest capacity={min_caps}")
            self.minp = dp_list[i]
        self.R1frac = self.minp*self.R1frac_ratio
        for c in self.clients:
            c.p = 1.0
            c.R2frac = self.R2frac
            c.R1frac = self.R1frac
            while c.p**2>c._capacity:
                c.p-=self.minp
            if c.p<1e-4: raise RuntimeError(f"minp={self.minp} cannot support training for clients with capacity={c._capacity}. Try to further reduce the value of minp by the factor 2.")
            self.client_ps.append(c.p)
        self.per_dicts = {}
        self.pset = set(self.client_ps)
        for p in self.pset:
            tmp = self._model_class.Model(p, R2frac=self.R2frac, R1frac=self.R1frac)
            tmp_dict = {k:v for k,v in tmp.state_dict().items() if 'filter_bank' not in k}
            self.per_dicts[p] = tmp_dict
        tmp = self._model_class.Model(1.0, R2frac=self.R2frac, R1frac=self.R1frac)
        self.shared_dict = {k: v for k, v in tmp.state_dict().items() if 'filter_bank' in k}

    def pack(self, client_id, mtype=0, *args, **kwargs):
        p = self.client_ps[client_id]
        res = copy.deepcopy(self.per_dicts[p])
        res.update(self.shared_dict)
        return {'md':res}

    def iterate(self):
        self.selected_clients = self.sample()
        mds = self.communicate(self.selected_clients)['md']
        # group clients
        shared_parts = [{k:v for k,v in md.items()  if 'filter_bank' in k} for md in mds]
        self.shared_dict = fuf._modeldict_weighted_average(shared_parts)
        per_parts = [{k:v for k,v in md.items()  if 'filter_bank' not in k} for md in mds]
        tmp_pclients = {p:[] for p in self.pset}
        for cid, pmd in zip(self.selected_clients, per_parts):
            pi = self.client_ps[cid]
            tmp_pclients[pi].append(pmd)
        for p in tmp_pclients:
            if len(tmp_pclients[p])>0:
                self.per_dicts[p] = fuf._modeldict_weighted_average(tmp_pclients[p])
        # set test model
        for p in self.per_dicts:
            self.per_dicts[p].update(self.shared_dict)
        for cid in range(self.num_clients):
            self.clients[cid].model.load_state_dict(self.per_dicts[self.client_ps[cid]])
        self.per_dicts = {p:{k:v for k,v in pv.items() if 'filter_bank' not in k} for p,pv in self.per_dicts.items()}
        return

class Client(fedavg.Client):
    def initialize(self, *args, **kwargs):
        self.model = self._model_class.Model(self.p, self.R2frac, self.R1frac)

    def unpack(self, received_pkg):
        self.model.load_state_dict(received_pkg['md'])
        return self.model

    def pack(self, model, *args, **kwargs):
        return {'md': self.model.state_dict()}

    def orth_loss(self, model):
        loss_fun = nn.MSELoss()
        loss = 0
        for n,p in model.named_parameters():
            if 'filter_bank' in n:
                all_bank = p
                num_all_bank = p.shape[0]
                B = all_bank.view(num_all_bank, -1)
                D = torch.mm(B, torch.t(B))
                D = loss_fun(D, torch.eye(num_all_bank, num_all_bank).to(p.device))
                loss = loss + D
        return loss

    @fuf.with_multi_gpus
    def train(self, model):
        r"""
        Standard local training procedure. Train the transmitted model with
        local training dataset.

        Args:
            model (FModule): the global model
        """
        model.train()
        optimizer = self.calculator.get_optimizer(model, lr=self.learning_rate, weight_decay=self.weight_decay,
                                                  momentum=self.momentum)
        for iter in range(self.num_steps):
            # get a batch of data
            batch_data = self.get_batch_data()
            model.zero_grad()
            # calculate the loss of the model on batched dataset through task-specified calculator
            loss = self.calculator.compute_loss(model, batch_data)['loss']
            loss += self.lmbd*self.orth_loss(model)
            loss.backward()
            if self.clip_grad > 0: torch.nn.utils.clip_grad_norm_(parameters=model.parameters(),
                                                                  max_norm=self.clip_grad)
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

def default_conv(
        in_channels, out_channels, kernel_size=3, stride=1, bias=True, groups=1):

    m = nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), stride=stride, bias=bias, groups=groups
    )
    return m

def default_norm(in_channels):
    return nn.BatchNorm2d(in_channels)

def default_act():
    return nn.ReLU(False)

class conv_basis(nn.Module):
    def __init__(self, filter_bank, in_channels, basis_size, n_basis, kernel_size, stride=1, bias=True, padding=0):
        super(conv_basis, self).__init__()
        self.in_channels = in_channels
        self.n_basis = n_basis
        self.kernel_size = kernel_size
        self.basis_size = basis_size
        self.stride = stride
        self.group = in_channels // basis_size
        self.weight = filter_bank
        self.bias = nn.Parameter(torch.zeros(n_basis)) if bias else None
        self.padding = padding
        # print(stride)

    def forward(self, x):
        if self.group == 1:
            x = F.conv2d(input=x, weight=self.weight, bias=self.bias, stride=self.stride, padding=self.kernel_size//2)
        else:
            # print(self.weight.shape)
            x = torch.cat([F.conv2d(input=xi, weight=self.weight, bias=self.bias, stride=self.stride,
                                    padding=self.kernel_size//2)
                           for xi in torch.split(x, self.basis_size, dim=1)], dim=1)
        return x

    def __repr__(self):
        s = 'Conv_basis(in_channels={}, basis_size={}, group={}, n_basis={}, kernel_size={}, out_channel={})'.format(
            self.in_channels, self.basis_size, self.group, self.n_basis, self.kernel_size, self.group * self.n_basis)
        return s

class DecomBlock(nn.Module):
    def __init__(self, filter_bank, in_channels, out_channels, n_basis, basis_size, kernel_size,
                 stride=1, bias=False, padding=0, conv=default_conv, norm=default_norm, act=default_act):
        super(DecomBlock, self).__init__()
        group = in_channels // basis_size
        modules = [conv_basis(filter_bank, in_channels, basis_size, n_basis, kernel_size, stride, bias, padding)]
        modules.append(conv(group * n_basis, out_channels, kernel_size=1, stride=1, bias=bias))
        self.conv = nn.Sequential(*modules)

    def forward(self, x):
        return self.conv(x)

def get_decom_conv(in_channels, out_channels, kernel_size, stride=1,  bias=False, padding=0, p=1.0, R1frac=1.0, R2frac=0.25, keep_dim_in=False):
    num_in, num_out = in_channels, out_channels
    R2 = int(num_out * R2frac)
    R1 = num_in if keep_dim_in else int(num_in*R1frac)
    filter_bank = nn.Parameter(torch.empty(R2, R1, kernel_size, kernel_size))
    X = torch.empty(R2, R1, kernel_size, kernel_size)
    torch.nn.init.orthogonal(X)
    filter_bank.data = copy.deepcopy(X)
    conv = DecomBlock(filter_bank, num_in if keep_dim_in else int(num_in*p), int(num_out * p), R2, R1, kernel_size=kernel_size, bias=bias, stride=stride, padding=padding)
    return filter_bank, conv

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

    class FLANCResBlock(nn.Module):
        expansion: int = 1

        def __init__(
                self,
                inplanes: int,
                planes: int,
                stride: int = 1,
                downsample=None,
                groups: int = 1,
                base_width: int = 64,
                dilation: int = 1,
                norm_layer=None,
                p=1.0,
                R1frac=1.0,
                R2frac=0.25,
                keep_dim_in=False,
        ) -> None:
            super().__init__()
            if norm_layer is None:
                norm_layer = nn.BatchNorm2d
            if groups != 1 or base_width != 64:
                raise ValueError("BasicBlock only supports groups=1 and base_width=64")
            if dilation > 1:
                raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
            # Both self.conv1 and self.downsample layers downsample the input when stride != 1
            self.filter_bank_conv1, self.conv1 = get_decom_conv(inplanes, planes, 3, stride=stride, p=p, R1frac=R1frac, R2frac=R2frac)

            # self.conv1 = pdconv3x3(inplanes, planes, stride, minp=minp, p=p, is_input_layer=is_input_layer)
            self.bn1 = norm_layer(int(p * planes))
            self.relu = nn.ReLU(inplace=True)
            self.filter_bank_conv2, self.conv2 = get_decom_conv(planes, planes, 3,  p=p, R1frac=R1frac, R2frac=R2frac)

            self.bn2 = norm_layer(int(p * planes))
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

    class FLANCResBottleneck(nn.Module):

        expansion: int = 4

        def __init__(
                self,
                inplanes: int,
                planes: int,
                stride: int = 1,
                downsample=None,
                groups: int = 1,
                base_width: int = 64,
                dilation: int = 1,
                norm_layer=None,
                p=1.0,
                R1frac=1.0,
                R2frac=0.25,
                keep_dim_in=False,
        ) -> None:
            super().__init__()
            if norm_layer is None:
                norm_layer = nn.BatchNorm2d
            width = int(planes * (base_width / 64.0)) * groups
            # Both self.conv2 and self.downsample layers downsample the input when stride != 1
            self.filter_bank_conv1, self.conv1 = get_decom_conv(inplanes, width, 1, p=p, R1frac=R1frac, R2frac=R2frac, keep_dim_in=keep_dim_in)
            self.bn1 = norm_layer(int(p * width))
            self.filter_bank_conv2, self.conv2 = get_decom_conv(width, width, 3, stride=stride, p=p, R1frac=R1frac, R2frac=R2frac)
            self.bn2 = norm_layer(int(p * width))
            self.filter_bank_conv3, self.conv3 = get_decom_conv(width, planes * self.expansion, 1, p=p, R1frac=R1frac, R2frac=R2frac)
            self.bn3 = norm_layer(int(planes * p) * self.expansion)
            self.relu = nn.ReLU(inplace=True)
            self.downsample = downsample
            self.stride = stride

        def forward(self, x):
            identity = x

            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.bn2(out)
            out = self.relu(out)

            out = self.conv3(out)
            out = self.bn3(out)

            if self.downsample is not None:
                identity = self.downsample(x)

            out += identity
            out = self.relu(out)

            return out

    class FLANCResNetEncoder(nn.Module):
        def __init__(
                self,
                block,
                layers,
                zero_init_residual: bool = False,
                groups: int = 1,
                width_per_group: int = 64,
                replace_stride_with_dilation=None,
                norm_layer=None,
                p: float = 1.0,
                R1frac: float=1.0,
                R2frac: float=0.25,
        ):
            super().__init__()
            usub._log_api_usage_once(self)
            if norm_layer is None: norm_layer = nn.BatchNorm2d
            self._norm_layer = norm_layer
            self.p = p
            self.R1frac = R1frac
            self.R2frac = R2frac
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
            self.filter_bank0, self.conv1 = get_decom_conv(3, self.inplanes, 7, stride=2, padding=3, bias=False, p=p, R1frac=R1frac, R2frac=R2frac, keep_dim_in=True)
            self.bn1 = norm_layer(int(self.inplanes * p))
            self.relu = nn.ReLU(inplace=True)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            self.layer1 = self._make_layer(block, 64, layers[0])
            self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
            self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
            self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

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
                    if isinstance(m, CIFAR10Model.FLANCResBottleneck) and m.bn3.weight is not None:
                        nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                    elif isinstance(m, CIFAR10Model.FLANCResBlock) and m.bn2.weight is not None:
                        nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

        def _make_layer(
                self,
                block,
                planes: int,
                blocks: int,
                stride: int = 1,
                dilate: bool = False,
        ):
            norm_layer = self._norm_layer
            downsample = None
            previous_dilation = self.dilation
            if dilate:
                self.dilation *= stride
                stride = 1
            if stride != 1 or self.inplanes != planes * block.expansion:
                self.filter_bank_down, conv_down = get_decom_conv(self.inplanes, planes * block.expansion, 1, stride=stride, bias=False, p=self.p, R1frac=self.R1frac, R2frac=self.R2frac)
                downsample = nn.Sequential(
                    conv_down,
                    norm_layer(int(planes * block.expansion * self.p)),
                )

            layers = []
            layers.append(
                block(
                    self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation,
                    norm_layer, p=self.p, R1frac=self.R1frac, R2frac=self.R2frac
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
                        p=self.p, R1frac=self.R1frac, R2frac=self.R2frac
                    )
                )
            return nn.Sequential(*layers)

        def _forward_impl(self, x):
            # See note [TorchScript super()]
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
            return x

        def forward(self, x):
            return self._forward_impl(x)

    class Model(fuf.FModule):
        def __init__(self, p:float=1.0, R2frac = 0.25, R1frac=0.0625):
            super().__init__()
            self.R1frac = R1frac
            self.R2frac = R2frac
            self.p = p
            self.encoder = CIFAR10Model.FLANCResNetEncoder(CIFAR10Model.FLANCResBlock, [2,2,2,2], norm_layer=lambda x: nn.GroupNorm(2, x), p=p, R1frac=R1frac, R2frac=R2frac)
            self.head = usub.PLinear(512 * CIFAR10Model.FLANCResBlock.expansion, 10, p=p, keep_dim_out=True)

        def forward(self, x):
            x = self.encoder(x)
            x = self.head(x)
            return x

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

    @classmethod
    def init_global_module(cls, object):
        if 'Server' in object.__class__.__name__:
            if not hasattr(object, '_model_class'):
                object._model_class = cls
                return

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

    class Model(fuf.FModule):
        def __init__(self, p:float=1.0, R2frac = 0.25, R1frac=0.0625):
            super().__init__()
            self.R1frac = R1frac
            self.R2frac = R2frac
            self.p = p
            self.filter_bank_1, self.conv1 = get_decom_conv(3, 64, 5, p=p, R1frac=R1frac, R2frac=R2frac, keep_dim_in=True)
            self.filter_bank_2, self.conv2 = get_decom_conv(64, 64, 5, p=p, R1frac=R1frac, R2frac=R2frac)
            self.filter_bank_3, self.fc1 = get_decom_conv(1600, 64, 5, p=p, R1frac=R1frac, R2frac=R2frac)
            self.filter_bank_4, self.fc2 = get_decom_conv(384, 192, 1, p=p, R1frac=R1frac, R2frac=R2frac)
            self.head = usub.PLinear(192, 100, keep_dim_out=True, p=p)
            self.relu = nn.ReLU()
            self.pool = nn.MaxPool2d(2)
            self.flatten = nn.Flatten(1)

        def forward(self, x):
            x = self.pool(self.relu(self.conv1(x)))
            x = self.pool(self.relu(self.conv2(x)))
            x = self.flatten(x)
            x = x.unsqueeze(-1).unsqueeze(-1)
            x = self.relu(self.fc1(x))
            x = self.relu(self.fc2(x))
            x = x.squeeze(-1).squeeze(-1)
            x = self.head(x)
            return x

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

    @classmethod
    def init_global_module(cls, object):
        if 'Server' in object.__class__.__name__:
            if not hasattr(object, '_model_class'):
                object._model_class = cls
                return

class DOMAINNETModel:
    class Model(fuf.FModule):
        """
        used for DomainNet and Office-Caltech10
        """

        def __init__(self, p:float=1.0, R2frac = 0.25, R1frac=0.0625):
            super().__init__()
            self.p = p
            self.relu = nn.ReLU(inplace=True)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2)
            self.filter_bank_1, self.conv1 = get_decom_conv(3, 64, kernel_size=11, stride=4, padding=2,  p=p, R1frac=R1frac, R2frac=R2frac, keep_dim_in=True)
            self.filter_bank_2, self.conv2 = get_decom_conv(64, 192, kernel_size=5, padding=2, p=p, R1frac=R1frac, R2frac=R2frac)
            self.filter_bank_3, self.conv3 = get_decom_conv(192, 384, kernel_size=3, padding=1, p=p, R1frac=R1frac, R2frac=R2frac)
            self.filter_bank_4, self.conv4 = get_decom_conv(384, 256, kernel_size=3, padding=1, p=p, R1frac=R1frac, R2frac=R2frac)
            self.filter_bank_5, self.conv5 = get_decom_conv(256, 256, kernel_size=3, padding=1, p=p, R1frac=R1frac, R2frac=R2frac)
            self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
            self.filter_bank_6, self.fc1 = get_decom_conv(256 * 6 * 6, 1024, kernel_size=1, p=p, R1frac=R1frac, R2frac=R2frac)
            self.filter_bank_7, self.fc2 = get_decom_conv(1024, 1024, kernel_size=1, p=p, R1frac=R1frac, R2frac=R2frac)
            self.head = usub.PLinear(1024, 10, keep_dim_out=True)

        def encoder(self, x):
            x = self.maxpool(self.relu(self.conv1(x)))
            x = self.maxpool(self.relu(self.conv2(x)))
            x = self.relu(self.conv3(x))
            x = self.relu(self.conv4(x))
            x = self.maxpool(self.relu(self.conv5(x)))
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.relu(self.fc1(x))
            x = self.relu(self.fc2(x))
            return x

        def forward(self, x):
            out = self.encoder(x)
            out = self.head(out)
            return out

    @classmethod
    def init_dataset(cls, object):
        pass

    @classmethod
    def init_local_module(cls, object):
        if 'Client' in object.__class__.__name__:
            if not hasattr(object, '_model_class'):
                object._model_class = cls
                return

    @classmethod
    def init_global_module(cls, object):
        if 'Server' in object.__class__.__name__:
            if not hasattr(object, '_model_class'):
                object._model_class = cls
                return
            else:
                if hasattr(object, '_track'):
                    return cls.Model(1.0)
                else:
                    return cls.Model(1.0)

class MNISTModel:
    class Model(fuf.FModule):
        def __init__(self, p:float=1.0, R2frac = 0.25, R1frac=0.0625):
            super().__init__()
            self.R1frac = R1frac
            self.R2frac = R2frac
            self.p = p
            num_in = 1
            num_out = 32
            kernel_size = 5
            R2 = int(num_out*R2frac)
            R1 = num_in
            self.filter_bank_1 = nn.Parameter(torch.empty(R2, R1, kernel_size, kernel_size))
            X = torch.empty(R2, R1, kernel_size, kernel_size)
            torch.nn.init.orthogonal(X)
            self.filter_bank_1.data = copy.deepcopy(X)
            self.conv1 = DecomBlock(self.filter_bank_1, num_in, int(num_out*p), R2, R1, kernel_size=kernel_size, bias=False, padding=2)

            num_in = 32
            num_out = 64
            kernel_size = 5
            R2 = int(num_out*R2frac)
            R1 = int(num_in*R1frac)
            self.filter_bank_2 = nn.Parameter(torch.empty(R2, R1, kernel_size, kernel_size))
            X = torch.empty(R2, R1, kernel_size, kernel_size)
            torch.nn.init.orthogonal(X)
            self.filter_bank_2.data = copy.deepcopy(X)
            self.conv2 = DecomBlock(self.filter_bank_2, int(num_in*p), int(num_out*p), R2, R1, kernel_size=kernel_size, bias=False, padding=2)

            num_in = 3136
            num_out = 512
            kernel_size = 1
            R2 = int(num_out*R2frac)
            R1 = int(num_in*R1frac)
            self.filter_bank_3 = nn.Parameter(torch.empty(R2, R1, kernel_size, kernel_size))
            X = torch.empty(R2, R1, kernel_size, kernel_size)
            torch.nn.init.orthogonal(X)
            self.filter_bank_3.data = copy.deepcopy(X)
            self.fc1 = DecomBlock(self.filter_bank_3, int(num_in*p), int(num_out*p), R2, R1, kernel_size=kernel_size, bias=False,)

            num_in = 512
            num_out = 128
            kernel_size = 1
            R2 = int(num_out*R2frac)
            R1 = int(num_in*R1frac)
            self.filter_bank_4 = nn.Parameter(torch.empty(R2, R1, kernel_size, kernel_size))
            X = torch.empty(R2, R1, kernel_size, kernel_size)
            torch.nn.init.orthogonal(X)
            self.filter_bank_4.data = copy.deepcopy(X)
            self.fc2 = DecomBlock(self.filter_bank_4, int(num_in*p), int(num_out*p), R2, R1, kernel_size=kernel_size, bias=False,)

            self.head = nn.Linear(int(num_out*p), 10)
            self.relu = nn.ReLU()
            self.pool = nn.MaxPool2d(2)
            self.flatten = nn.Flatten(1)

        def forward(self, x):
            x = self.pool(self.relu(self.conv1(x)))
            x = self.pool(self.relu(self.conv2(x)))
            x = self.flatten(x)
            x = x.unsqueeze(-1).unsqueeze(-1)
            x = self.relu(self.fc1(x))
            x = self.relu(self.fc2(x))
            x = x.squeeze(-1).squeeze(-1)
            x = self.head(x)
            return x

    @classmethod
    def init_dataset(cls, object):
        pass

    @classmethod
    def init_local_module(cls, object):
        if 'Client' in object.__class__.__name__:
            if not hasattr(object, '_model_class'):
                object._model_class = cls
                return

    @classmethod
    def init_global_module(cls, object):
        if 'Server' in object.__class__.__name__:
            if not hasattr(object, '_model_class'):
                object._model_class = cls
                return

class FASHIONModel:
    class Model(fuf.FModule):
        def __init__(self, p:float=1.0, R2frac = 0.25, R1frac=0.0625):
            super().__init__()
            self.R1frac = R1frac
            self.R2frac = R2frac
            self.p = p
            num_in = 1
            num_out = 32
            kernel_size = 5
            R2 = int(num_out*R2frac)
            R1 = num_in
            self.filter_bank_1 = nn.Parameter(torch.empty(R2, R1, kernel_size, kernel_size))
            X = torch.empty(R2, R1, kernel_size, kernel_size)
            torch.nn.init.orthogonal_(X)
            self.filter_bank_1.data = copy.deepcopy(X)
            self.conv1 = DecomBlock(self.filter_bank_1, num_in, int(num_out*p), R2, R1, kernel_size=kernel_size, bias=False, padding=2)

            num_in = 32
            num_out = 64
            kernel_size = 5
            R2 = int(num_out*R2frac)
            R1 = int(num_in*R1frac)
            self.filter_bank_2 = nn.Parameter(torch.empty(R2, R1, kernel_size, kernel_size))
            X = torch.empty(R2, R1, kernel_size, kernel_size)
            torch.nn.init.orthogonal(X)
            self.filter_bank_2.data = copy.deepcopy(X)
            self.conv2 = DecomBlock(self.filter_bank_2, int(num_in*p), int(num_out*p), R2, R1, kernel_size=kernel_size, bias=False, padding=2)

            num_in = 3136
            num_out = 512
            kernel_size = 1
            R2 = int(num_out*R2frac)
            R1 = int(num_in*R1frac)
            self.filter_bank_3 = nn.Parameter(torch.empty(R2, R1, kernel_size, kernel_size))
            X = torch.empty(R2, R1, kernel_size, kernel_size)
            torch.nn.init.orthogonal(X)
            self.filter_bank_3.data = copy.deepcopy(X)
            self.fc1 = DecomBlock(self.filter_bank_3, int(num_in*p), int(num_out*p), R2, R1, kernel_size=kernel_size, bias=False,)

            num_in = 512
            num_out = 128
            kernel_size = 1
            R2 = int(num_out*R2frac)
            R1 = int(num_in*R1frac)
            self.filter_bank_4 = nn.Parameter(torch.empty(R2, R1, kernel_size, kernel_size))
            X = torch.empty(R2, R1, kernel_size, kernel_size)
            torch.nn.init.orthogonal(X)
            self.filter_bank_4.data = copy.deepcopy(X)
            self.fc2 = DecomBlock(self.filter_bank_4, int(num_in*p), int(num_out*p), R2, R1, kernel_size=kernel_size, bias=False,)

            self.head = nn.Linear(int(num_out*p), 10)
            self.relu = nn.ReLU()
            self.pool = nn.MaxPool2d(2)
            self.flatten = nn.Flatten(1)

        def forward(self, x):
            x = self.pool(self.relu(self.conv1(x)))
            x = self.pool(self.relu(self.conv2(x)))
            x = self.flatten(x)
            x = x.unsqueeze(-1).unsqueeze(-1)
            x = self.relu(self.fc1(x))
            x = self.relu(self.fc2(x))
            x = x.squeeze(-1).squeeze(-1)
            x = self.head(x)
            return x

    @classmethod
    def init_dataset(cls, object):
        pass

    @classmethod
    def init_local_module(cls, object):
        if 'Client' in object.__class__.__name__:
            if not hasattr(object, '_model_class'):
                object._model_class = cls
                return

    @classmethod
    def init_global_module(cls, object):
        if 'Server' in object.__class__.__name__:
            if not hasattr(object, '_model_class'):
                object._model_class = cls
                return