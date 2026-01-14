import copy
import os
from collections import OrderedDict
import torch
import numpy as np
import flgo.algorithm.fedavg as fedavg
import flgo.utils.fmodule as fuf
from torch.utils.data import Dataset
from torchvision.transforms import RandomCrop, RandomHorizontalFlip
import torchvision
import torch.nn as nn
import flgo.utils.submodule as usub
class Server(fedavg.Server):
    def initialize(self, *args, **kwargs):
        self.init_algo_para({'rate':0.5, 'use_label_mask':True})
        self.rp = 1.0
        self._track = False
        self.model = self._model_class.init_global_module(self)
        self.client_capacity = [c._capacity for c in self.clients]
        for c in self.clients:
            c.pi = 0
            while self.rate**(2*c.pi)>c._capacity:
                c.pi+=1
        self.client_level = [c.pi for c in self.clients]
        self.level_set = set(sorted(self.client_level))
        self.model_shapes = {}
        for pi in range(max(self.level_set)+1):
            tmp_md = self._model_class.Model(self.rate**pi).state_dict()
            self.model_shapes[pi] = {k:v.shape for k,v in tmp_md.items()}
        # only for testing
        self.track_model_shapes = {}
        self.test_model_set = {}
        for pi in self.level_set:
            self.test_model_set[pi] = self._model_class.Model(self.rate**pi, track=True).to(self.device)
            self.track_model_shapes[pi] = {k:v.shape for k,v in self.test_model_set[pi].state_dict().items()}
        self.test_model = self._model_class.Model(1.0, track=True).to(self.device)
        md = self.test_model.state_dict()
        md.update(self.model.state_dict())
        self.test_model.load_state_dict(md)
        self.distribute_test_model()
        all_train_data = torch.utils.data.ConcatDataset([c.train_data for c in self.clients])
        self.train_loader = torch.utils.data.DataLoader(all_train_data, batch_size=self.clients[0].batch_size)

    def generate_sub_state_dict(self, model, level_set, shapes):
        md = model.state_dict()
        res = {}
        for pi in level_set:
            tmp = {}
            for k in md:
                v = md[k]
                for dim, l in enumerate(shapes[pi][k]):
                    v = v.narrow(dim, 0, l)
                tmp[k] = v
            res[pi] = tmp
        return res

    def pack(self, client_id, mtype=0, *args, **kwargs):
        pi = self.client_level[client_id]
        return {'w': self.crt_dict_set[pi]}

    def aggregate(self, models: list, label_masks = None):
        mds = [mi.state_dict() for mi in models]
        full_model_shape = self.model_shapes[0]
        tmp_md = {k: torch.zeros(s) for k, s in full_model_shape.items()}
        mask = {k: torch.zeros(s) for k, s in full_model_shape.items()}
        for i,md in enumerate(mds):
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

                if not self.use_label_mask or 'head' not in k:
                    m = torch.ones_like(v)
                else:
                    lb_mask = label_masks[i]
                    m = torch.ones_like(v)
                    for lb in range(len(lb_mask)):
                        if lb_mask[lb] == 0:
                            v[lb] = 0
                            m[lb] = 0

                target_weight += v
                target_mask += m
        for k in tmp_md.keys():
            tmp_md[k]/=(mask[k]+1e-8)
        self.model.load_state_dict(tmp_md)
        return self.model

    def iterate(self):
        self.selected_clients = self.sample()
        self.crt_dict_set = self.generate_sub_state_dict(self.model, set(self.client_level[cid] for cid in self.selected_clients), self.model_shapes )
        res = self.communicate(self.selected_clients, mtype=0)
        self.model = self.aggregate(res['model'], res['label_mask'])
        tmp_md = self.test_model.state_dict()
        tmp_md.update(self.model.state_dict())
        self.test_model.load_state_dict(tmp_md)
        # collect bn stats
        self.test_model.train()
        with torch.no_grad():
            for i, batch_data in enumerate(self.train_loader):
                self.test_model(batch_data[0].to(self.device))
        self.distribute_test_model()

    def distribute_test_model(self):
        test_dicts = self.generate_sub_state_dict(self.test_model, self.level_set, self.track_model_shapes)
        for pi in test_dicts:
            self.test_model_set[pi].load_state_dict(test_dicts[pi])
        for cid in range(self.num_clients):
            self.clients[cid].model = self.test_model_set[self.client_level[cid]]

class Client(fedavg.Client):
    def initialize(self, *args, **kwargs):
        self.p = self.rate**self.pi
        self.train_model = self._model_class.Model(self.p)
        self.actions = {0:self.reply, 1:self.set_test_model}
        local_labels = list(set([d[-1] for d in self.train_data]))
        self.label_mask = torch.zeros(self.model.num_classes).to(self.device)
        self.label_mask[local_labels] = 1

    def unpack(self, received_pkg):
        self.train_model.load_state_dict(received_pkg['w'])
        return self.train_model

    def pack(self, model, *args, **kwargs):
        return {'model':model, 'label_mask':self.label_mask.cpu()}

    def set_test_model(self, pkg):
        self.model.load_state_dict(pkg['w'])

    @fuf.with_multi_gpus
    def train(self, model):
        model.train()
        optimizer = self.calculator.get_optimizer(model, lr=self.learning_rate, weight_decay=self.weight_decay,
                                                  momentum=self.momentum)
        for iter in range(self.num_steps):
            # get a batch of data
            batch_data = self.get_batch_data()
            batch_data = self.calculator.to_device(batch_data)
            model.zero_grad()
            y = model(batch_data[0])
            if self.use_label_mask:
                y = y.masked_fill(self.label_mask==0, 0)
            loss = self.calculator.criterion(y, batch_data[-1])
            loss.backward()
            if self.clip_grad > 0: torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=self.clip_grad)
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

class Scaler(nn.Module):
    def __init__(self, rate):
        super().__init__()
        self.rate = rate

    def forward(self, input):
        output = input / self.rate if self.training else input
        return output

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

    class SPBasicBlock(usub.rn.BasicBlock):
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
                p: float = 1.0,
                keep_dim_in: bool = False,
                keep_dim_out: bool = False,
        ):
            super().__init__(
                inplanes=inplanes if keep_dim_in else int(inplanes * p),
                planes=planes if keep_dim_out else int(planes * p),
                stride=stride,
                downsample=downsample,
                groups=groups,
                base_width=base_width,
                dilation=dilation,
                norm_layer=norm_layer,
            )
            self.p = p
            self.scaler = Scaler(p)

        def forward(self, x):
            identity = x
            out = self.conv1(x)
            out = self.scaler(out)
            out = self.bn1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.scaler(out)
            out = self.bn2(out)

            if self.downsample is not None:
                identity = self.downsample(x)

            out += identity
            out = self.relu(out)

            return out

    class SPBottleneck(usub.rn.Bottleneck):
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
                p: float = 1.0,
                keep_dim_in: bool = False,
                keep_dim_out: bool = False,
        ) -> None:
            super().__init__(
                inplanes=inplanes if keep_dim_in else int(inplanes * p),
                planes=planes if keep_dim_out else int(planes * p),
                stride=stride,
                downsample=downsample,
                groups=groups,
                base_width=base_width,
                dilation=dilation,
                norm_layer=norm_layer,
            )
            self.p = p
            self.scaler = Scaler(self.p)

        def forward(self, x):
            identity = x

            out = self.conv1(x)
            out = self.scaler(out)
            out = self.bn1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.scaler(out)
            out = self.bn2(out)
            out = self.relu(out)

            out = self.conv3(out)
            out = self.scaler(out)
            out = self.bn3(out)

            if self.downsample is not None:
                identity = self.downsample(x)

            out += identity
            out = self.relu(out)

            return out

    class SPResNetEncoder(nn.Module):
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
        ):
            super().__init__()
            usub._log_api_usage_once(self)
            if norm_layer is None: norm_layer = nn.BatchNorm2d
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
            self.scaler = Scaler(self.p)
            self.base_width = width_per_group
            self.conv1 = usub.PConv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False,
                                      keep_dim_in=True, p=p)
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
                    if isinstance(m, usub.PBottleneck) and m.bn3.weight is not None:
                        nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                    elif isinstance(m, usub.PBasicBlock) and m.bn2.weight is not None:
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
                downsample = nn.Sequential(
                    usub.rn.conv1x1(int(self.inplanes * self.p), int(planes * block.expansion * self.p), stride),
                    norm_layer(int(planes * block.expansion * self.p)),
                )

            layers = []
            layers.append(
                block(
                    self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation,
                    norm_layer, p=self.p
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
                        p=self.p
                    )
                )
            return nn.Sequential(*layers)

        def _forward_impl(self, x):
            # See note [TorchScript super()]
            x = self.conv1(x)
            x = self.scaler(x)
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
        def __init__(self, p: float = 1.0, track=False):
            super().__init__()
            self.num_classes = 10
            self.p = p
            self.encoder = CIFAR10Model.SPResNetEncoder(CIFAR10Model.SPBasicBlock, [2,2,2,2], norm_layer=lambda x: nn.GroupNorm(2, x), p=p)
            self.head = usub.PLinear(512*CIFAR10Model.SPBasicBlock.expansion, 10, bias=True, p=p, keep_dim_out=True)

        def forward(self, x):
            x = self.encoder(x)
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
                return cls.Model(object.p ** object.p)

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
        def __init__(self, p: float = 1.0, track=False):
            super().__init__()
            self.num_classes = 100
            self.encoder = nn.Sequential(
                nn.Conv2d(3, int(64 * p), 5, bias=True),
                Scaler(p),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(int(64 * p), int(64 * p), 5, bias=True),
                Scaler(p),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Flatten(1),
                nn.Linear(int(1600 * p), int(384 * p), bias=True),
                Scaler(p),
                nn.ReLU(),
                nn.Linear(int(384 * p), int(192 * p), bias=True),
                Scaler(p),
                nn.ReLU(),
            )
            self.head = nn.Linear(int(192 * p), 100, bias=True)

        def forward(self, x):
            x = self.encoder(x)
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
                return cls.Model(object.p ** object.p)

    @classmethod
    def init_global_module(cls, object):
        if 'Server' in object.__class__.__name__:
            if not hasattr(object, '_model_class'):
                object._model_class = cls
                return
            else:
                if hasattr(object, '_track'):
                    return cls.Model(1.0, object._track)
                else:
                    return cls.Model(1.0)

class MNISTModel:
    class Model(fuf.FModule):
        def __init__(self, rate: float = 1.0, track=False):
            super().__init__()
            self.num_classes = 10
            self.encoder = nn.Sequential(
                nn.Conv2d(1, int(32*rate), 5, bias=True, padding=2),
                Scaler(rate),
                nn.BatchNorm2d(int(32*rate), momentum=None, track_running_stats=track),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(int(32*rate), int(64*rate), 5, bias=True, padding=2),
                Scaler(rate),
                nn.BatchNorm2d(int(64 * rate), momentum=None, track_running_stats=track),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Flatten(1),
                nn.Linear(int(3136*rate), int(512*rate), bias=True),
                Scaler(rate),
                nn.BatchNorm1d(int(512*rate), momentum=None, track_running_stats=track),
                nn.ReLU(),
                nn.Linear(int(512*rate), int(128*rate), bias=True),
                Scaler(rate),
                nn.BatchNorm1d(int(128 * rate), momentum=None, track_running_stats=track),
                nn.ReLU(),
            )
            self.head = nn.Linear(int(128*rate), 10, bias=True)

        def forward(self, x):
            x = self.encoder(x)
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
                return cls.Model(object.p ** object.p)

    @classmethod
    def init_global_module(cls, object):
        if 'Server' in object.__class__.__name__:
            if not hasattr(object, '_model_class'):
                object._model_class = cls
                return
            else:
                if hasattr(object, '_track'):
                    return cls.Model(1.0, object._track)
                else:
                    return cls.Model(1.0)

class FASHIONModel:
    class Model(fuf.FModule):
        def __init__(self, rate: float = 1.0, track=False):
            super().__init__()
            self.num_classes = 10
            self.encoder = nn.Sequential(
                nn.Conv2d(1, int(32*rate), 5, bias=True, padding=2),
                Scaler(rate),
                nn.BatchNorm2d(int(32*rate), momentum=None, track_running_stats=track),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(int(32*rate), int(64*rate), 5, bias=True, padding=2),
                Scaler(rate),
                nn.BatchNorm2d(int(64 * rate), momentum=None, track_running_stats=track),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Flatten(1),
                nn.Linear(int(3136*rate), int(512*rate), bias=True),
                Scaler(rate),
                nn.BatchNorm1d(int(512*rate), momentum=None, track_running_stats=track),
                nn.ReLU(),
                nn.Linear(int(512*rate), int(128*rate), bias=True),
                Scaler(rate),
                nn.BatchNorm1d(int(128 * rate), momentum=None, track_running_stats=track),
                nn.ReLU(),
            )
            self.head = nn.Linear(int(128*rate), 10, bias=True)

        def forward(self, x):
            x = self.encoder(x)
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
                return cls.Model(object.p ** object.p)

    @classmethod
    def init_global_module(cls, object):
        if 'Server' in object.__class__.__name__:
            if not hasattr(object, '_model_class'):
                object._model_class = cls
                return
            else:
                if hasattr(object, '_track'):
                    return cls.Model(1.0, object._track)
                else:
                    return cls.Model(1.0)

class DOMAINNETModel:
    class Model(fuf.FModule):
        def __init__(self, p: float = 1.0, track=False, num_classes=10):
            super().__init__()
            self.p = p
            self.num_classes = num_classes
            self.features = nn.Sequential(
                OrderedDict([
                    ('conv1',  usub.PConv2d(3, 64, 11, stride=4, padding=2, bias=True, p=p, keep_dim_in=True)),
                    ('scaler1', Scaler(p)),
                    ('relu1', nn.ReLU(inplace=True)),
                    ('maxpool1', nn.MaxPool2d(kernel_size=3, stride=2)),

                    ('conv2', usub.PConv2d(64, 192, 5, padding=2, bias=True, p=p)),
                    ('scaler2', Scaler(p)),
                    ('relu2', nn.ReLU(inplace=True)),
                    ('maxpool2', nn.MaxPool2d(kernel_size=3, stride=2)),

                    ('conv3', usub.PConv2d(192, 384, 3, padding=1, bias=True, p=p)),
                    ('scaler3', Scaler(p)),
                    ('relu3', nn.ReLU(inplace=True)),

                    ('conv4', usub.PConv2d(384, 256, 3, padding=1, bias=True, p=p)),
                    ('scaler4', Scaler(p)),
                    ('relu4', nn.ReLU(inplace=True)),

                    ('conv5', usub.PConv2d(256, 256, 3, padding=1, bias=True, p=p)),
                    ('scaler5', Scaler(p)),
                    ('relu5', nn.ReLU(inplace=True)),
                    ('maxpool5', nn.MaxPool2d(kernel_size=3, stride=2)),
                ])
            )
            self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
            self.fc1 = usub.PLinear(256 * 6 * 6, 1024, bias=True, p=p)
            self.scaler6 = Scaler(p)
            self.relu = nn.ReLU()
            self.fc2 = usub.PLinear(1024, 1024, bias=True, p=p)
            self.scaler7 = Scaler(p)
            self.head = usub.PLinear(1024, num_classes, bias=True, p=p, keep_dim_out=True)

        def encoder(self, x):
            x = self.features(x)
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc1(x)
            x = self.scaler6(x)
            x = self.relu(x)
            x = self.fc2(x)
            x = self.scaler7(x)
            x = self.relu(x)
            return x

        def forward(self, x):
            x = self.encoder(x)
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
            else:
                return cls.Model(object.p ** object.p)

    @classmethod
    def init_global_module(cls, object):
        if 'Server' in object.__class__.__name__:
            if not hasattr(object, '_model_class'):
                object._model_class = cls
                return
            else:
                if hasattr(object, '_track'):
                    return cls.Model(1.0, object._track)
                else:
                    return cls.Model(1.0)

