import copy
import os
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch
import flgo.algorithm.fedavg as fedavg
import flgo.utils.fmodule as fuf
from torch.utils.data import Dataset
from torchvision.transforms import RandomCrop, RandomHorizontalFlip
import torchvision
import flgo.utils.submodule as usub

class Server(fedavg.Server):
    def initialize(self, *args, **kwargs):
        self.init_algo_para({'kd':True, 'dp':0.0625})
        self.model = self._model_class.Model().to(self.device)
        for c in self.clients:
            c.p = 1.0
            while (c.p ** 2 > c._capacity):
                c.p -= self.dp
        self.client_pmax = [c.p for c in self.clients]
        self.pmax_set = set(sorted(self.client_pmax))
        self.pmax_set.add(1.0)
        self.pmodel_shapes = {}
        for p in self.pmax_set:
            tmp_md = self._model_class.Model(p).state_dict()
            self.pmodel_shapes[p] = {k:v.shape for k,v in tmp_md.items()}


    def pack(self, client_id, mtype=0, *args, **kwargs):
        return {'w': self.generate_subdict(self.client_pmax[client_id])}

    def iterate(self):
        self.selected_clients = self.sample()
        models = self.communicate(self.selected_clients)['model']
        self.model = self.aggregate(models)
        # set models for test
        pdicts = {}
        for pi in self.pmax_set:
            pdicts[pi] = self.generate_subdict(pi)
        for c in self.clients:
            pc = c.p
            c.model.load_state_dict(pdicts[pc])

    def generate_subdict(self, p):
        md = copy.deepcopy(self.model.state_dict())
        layer_shapes = self.pmodel_shapes[p]
        for k in md.keys():
            lshape = layer_shapes[k]
            for dim, l in enumerate(lshape):
                md[k] = md[k].narrow(dim, 0, l)
        return md

    def aggregate(self, models: list, *args, **kwargs):
        mds = [mi.state_dict() for mi in models]
        full_model_shape = self.pmodel_shapes[1.0]
        tmp_md = {k: torch.zeros(s) for k, s in full_model_shape.items()}
        mask = {k: torch.zeros(s) for k, s in full_model_shape.items()}
        for i, md in enumerate(mds):
            for k, v in md.items():
                s = v.shape
                cmd_md = 'tmp_md[k]['
                for d in s:
                    cmd_md += f':{d},'
                cmd_md = cmd_md[:-1]
                cmd_md += ']'
                target_weight = eval(cmd_md)

                cmd_mask = 'mask[k]['
                for d in s:
                    cmd_mask += f':{d},'
                cmd_mask = cmd_mask[:-1]
                cmd_mask += ']'
                target_mask = eval(cmd_mask)

                target_weight += v.to(target_weight.device)
                target_mask += torch.ones_like(v).to(target_weight.device)
        for k in tmp_md.keys():
            tmp_md[k] /= (mask[k] + 1e-8)
        self.model.load_state_dict(tmp_md)
        return self.model

class Client(fedavg.Client):
    def initialize(self, *args, **kwargs):
        self.model = self._model_class.Model(self.p).to(self.device)
        self.ps = [self.dp * i / self.p for i in range(1, int(self.p / self.dp))]

    def unpack(self, received_pkg):
        w = received_pkg['w']
        self.model.load_state_dict(w)
        return self.model

    @fuf.with_multi_gpus
    def train(self, model):
        model.train()
        optimizer = self.calculator.get_optimizer(model, lr=self.learning_rate, weight_decay=self.weight_decay,
                                                  momentum=self.momentum)
        for iter in range(self.num_steps):
            model.zero_grad()
            batch_data = self.get_batch_data()
            batch_data = self.calculator.to_device(batch_data)
            target = batch_data[-1]
            y = model(batch_data[0], np.random.choice(self.ps) if len(self.ps)>0 else 1.0)
            loss = 0.0
            if self.kd:
                yfull = model(batch_data[0])
                loss += self.calculator.criterion(yfull, target)
                target = yfull.detach().softmax(dim=1)
            loss += self.calculator.criterion(y, target)
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

class ODConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super(ODConv2d, self).__init__(*args, **kwargs)

    def forward(self, x, p=None):
        in_dim = x.size(1)  # second dimension is input dimension
        if not p:  # i.e., don't apply OD
            out_dim = self.out_channels
        else:
            out_dim = int(np.ceil(self.out_channels * p))
        # subsampled weights and bias
        weights_red = self.weight[:out_dim, :in_dim]
        bias_red = self.bias[:out_dim] if self.bias is not None else None
        return self._conv_forward(x, weights_red, bias_red)

class ODLinear(nn.Linear):
    def __init__(self, *args, **kwargs):
        super(ODLinear, self).__init__(*args, **kwargs)

    def forward(self, x, p=None):
        in_dim = x.size(1)  # second dimension is input dimension
        if not p:  # i.e., don't apply OD
            out_dim = self.out_features
        else:
            out_dim = int(np.ceil(self.out_features * p))
        # subsampled weights and bias
        weights_red = self.weight[:out_dim, :in_dim]
        bias_red = self.bias[:out_dim] if self.bias is not None else None
        return F.linear(x, weights_red, bias_red)

class ODGroupNorm(nn.GroupNorm):
    def __init__(self, *args, **kwargs):
        super(ODGroupNorm, self).__init__(*args, **kwargs)

    def forward(self, x, p=None):
        if not p:  # i.e., don't apply OD
            out_dim = self.num_channels
        else:
            out_dim = int(np.ceil(self.num_channels * p))
        weights_red = self.weight[:out_dim]
        bias_red = self.bias[:out_dim] if self.bias is not None else None
        # subsampled weights and bias
        return F.group_norm(x, self.num_groups, weights_red, bias_red, self.eps)

class ODBasicBlock(nn.Module):
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
        self.conv1 = ODConv2d(in_channels=inplanes, out_channels=planes, kernel_size=3, stride=stride, padding=dilation, dilation=dilation, groups=groups)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = ODConv2d(in_channels=planes, out_channels=planes, kernel_size=3, padding=1)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x, p=None):
        in_dim = x.size(1)
        identity = x
        out = self.conv1(x, p)
        out = self.bn1(out, p)
        out = self.relu(out)
        out = self.conv2(out, p)
        out = self.bn2(out, p)
        if self.downsample is not None:
            identity = self.downsample[0](x, p)
            identity = self.downsample[1](identity, p)
        out += identity
        out = self.relu(out)
        return out

class ODBottleneck(nn.Module):
    expansion: int = 4

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
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = ODConv2d(inplanes, width, 1)
        self.bn1 = norm_layer(width)
        self.conv2 = ODConv2d(width, width,3, stride=stride, groups=groups, dilation=dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = ODConv2d(width, planes * self.expansion, 1)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x, p=None):
        identity = x

        out = self.conv1(x, p)
        out = self.bn1(out, p)
        out = self.relu(out)

        out = self.conv2(out, p)
        out = self.bn2(out, p)
        out = self.relu(out)

        out = self.conv3(out, p)
        out = self.bn3(out, p)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ODResNet(fuf.FModule):
    def __init__(
        self,
        block,
        layers,
        num_classes: int = 10,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation= None,
        norm_layer = None,
        p = 1.0,
    ) -> None:
        super().__init__()
        usub._log_api_usage_once(self)
        if norm_layer is None:
            norm_layer = ODGroupNorm
        self._norm_layer = norm_layer
        self.p = p
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
        self.conv1 = ODConv2d(3, int(self.inplanes*p), kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(int(self.inplanes*p))
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = ODLinear(int(512 * block.expansion *p), num_classes)

        for m in self.modules():
            if isinstance(m, ODConv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, ODGroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, ODBottleneck) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, ODBasicBlock) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

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
        if stride != 1 or int(self.inplanes*self.p) != int(planes*self.p) * block.expansion:
            downsample = nn.Sequential(
                ODConv2d(int(self.inplanes*self.p), int(planes*self.p) * block.expansion, 1, stride=stride),
                norm_layer(int(planes*self.p) * block.expansion),
            )

        layers = []
        layers.append(
            block(
                int(self.inplanes*self.p), int(planes*self.p), stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer
            )
        )
        self.inplanes = int(planes * block.expansion)
        for _ in range(1, blocks):
            layers.append(
                block(
                    int(self.inplanes*self.p),
                    int(planes*self.p),
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x, p=None):
        # See note [TorchScript super()]
        x = self.conv1(x, p)
        x = self.bn1(x, p)
        x = self.relu(x)
        x = self.maxpool(x)
        for i in range(4):
            layer = getattr(self, f'layer{i+1}')
            for li in layer:
                x = li(x, p)
        # x = self.layer1(x, p)
        # x = self.layer2(x, p)
        # x = self.layer3(x, p)
        # x = self.layer4(x, p)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.head(x)
        return x

    def forward(self, x, p=None):
        return self._forward_impl(x, p)

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

    class Model(fuf.FModule):
        def __init__(self, p: float = 1.0):
            super().__init__()
            self.p = p
            self.resnet = ODResNet(ODBasicBlock, [2,2,2,2], norm_layer=lambda x: ODGroupNorm(2, x), p=p, num_classes=10)

        def forward(self, x, p=None):
            return self.resnet(x, p)

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
                return cls.Model(object.p)

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
        def __init__(self, pmax: float = 1.0):
            super().__init__()
            self.num_classes = 10
            self.relu = nn.ReLU()
            self.pool2d = nn.MaxPool2d(2)
            self.conv1 = ODConv2d(3, int(64 * pmax), 5, bias=True)
            self.conv2 = ODConv2d(int(64 * pmax), int(64 * pmax), 5, bias=True)
            self.flatten = nn.Flatten(1)
            self.linear1 = ODLinear(int(1600 * pmax), int(384 * pmax), bias=True)
            self.linear2 = ODLinear(int(384 * pmax), int(192 * pmax), bias=True)
            self.head = ODLinear(int(192 * pmax), 100, bias=True)


        def encoder(self, x, p=None):
            x = self.pool2d(self.relu(self.conv1(x, p)))
            x = self.pool2d(self.relu(self.conv2(x, p)))
            x = self.flatten(x)
            x = self.relu(self.linear1(x, p))
            x = self.relu(self.linear2(x, p))
            return x

        def forward(self, x, p=None):
            x = self.encoder(x, p)
            return self.head(x)

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
                return cls.Model(object.p)

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
        def __init__(self, pmax: float = 1.0):
            super().__init__()
            self.num_classes = 10
            self.relu = nn.ReLU()
            self.pool2d = nn.MaxPool2d(2)
            self.conv1 = ODConv2d(1, int(32 * pmax), 5, bias=True, padding=2)
            self.conv2 = ODConv2d(int(32 * pmax), int(64 * pmax), 5, bias=True, padding=2)
            self.flatten = nn.Flatten(1)
            self.linear1 = ODLinear(int(3136 * pmax), int(512 * pmax), bias=True)
            self.linear2 = ODLinear(int(512 * pmax), int(128 * pmax), bias=True)
            self.head = ODLinear(int(128 * pmax), 10, bias=True)


        def encoder(self, x, p=None):
            x = self.pool2d(self.relu(self.conv1(x, p)))
            x = self.pool2d(self.relu(self.conv2(x, p)))
            x = self.flatten(x)
            x = self.relu(self.linear1(x, p))
            x = self.relu(self.linear2(x, p))
            return x

        def forward(self, x, p=None):
            x = self.encoder(x, p)
            return self.head(x)

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
                return cls.Model(object.p)

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

class FASHIONModel:
    class Model(fuf.FModule):
        def __init__(self, pmax: float = 1.0):
            super().__init__()
            self.num_classes = 10
            self.relu = nn.ReLU()
            self.pool2d = nn.MaxPool2d(2)
            self.conv1 = ODConv2d(1, int(32 * pmax), 5, bias=True, padding=2)
            self.conv2 = ODConv2d(int(32 * pmax), int(64 * pmax), 5, bias=True, padding=2)
            self.flatten = nn.Flatten(1)
            self.linear1 = ODLinear(int(3136 * pmax), int(512 * pmax), bias=True)
            self.linear2 = ODLinear(int(512 * pmax), int(128 * pmax), bias=True)
            self.head = ODLinear(int(128 * pmax), 10, bias=True)


        def encoder(self, x, p=None):
            x = self.pool2d(self.relu(self.conv1(x, p)))
            x = self.pool2d(self.relu(self.conv2(x, p)))
            x = self.flatten(x)
            x = self.relu(self.linear1(x, p))
            x = self.relu(self.linear2(x, p))
            return x

        def forward(self, x, p=None):
            x = self.encoder(x, p)
            return self.head(x)

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
                return cls.Model(object.p)

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

class DOMAINNETModel:
    class Model(fuf.FModule):
        """
        used for DomainNet and Office-Caltech10
        """

        def __init__(self, pmax:float=1.0, num_classes=10):
            super().__init__()
            self.pmax = pmax
            self.relu = nn.ReLU(inplace=True)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2)
            self.conv1 = ODConv2d(3, int(pmax*64), kernel_size=11, stride=4, padding=2)
            self.conv2 = ODConv2d(int(pmax*64), int(pmax*192), kernel_size=5, padding=2)
            self.conv3 = ODConv2d(int(pmax*192), int(pmax*384), kernel_size=3, padding=1)
            self.conv4 = ODConv2d(int(pmax*384), int(pmax*256), kernel_size=3, padding=1)
            self.conv5 = ODConv2d(int(pmax*256), int(pmax*256), kernel_size=3, padding=1)
            self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
            self.fc1 = ODLinear(int(pmax*256 * 6 * 6), int(pmax*1024))
            self.fc2 = ODLinear(int(pmax*1024), int(pmax*1024))
            self.head = ODLinear(int(pmax*1024), num_classes)

        def encoder(self, x, p=None):
            x = self.maxpool(self.relu(self.conv1(x, p)))
            x = self.maxpool(self.relu(self.conv2(x, p)))
            x = self.relu(self.conv3(x, p))
            x = self.relu(self.conv4(x, p))
            x = self.maxpool(self.relu(self.conv5(x, p)))
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.relu(self.fc1(x, p))
            x = self.relu(self.fc2(x, p))
            return x

        def forward(self, x, p=None):
            x = self.encoder(x, p)
            x = self.head(x)
            return x