import copy
import math
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
        self.init_algo_para({'dp':1.0/16})
        self.model = self._model_class.Model().to(self.device)
        for c in self.clients:
            c.p = 1.0
            while (c.p**2 > c._capacity):
                c.p -= self.dp
        self.client_p = [c.p for c in self.clients]
        self.p_set = set(sorted(self.client_p))
        self.pmodel_shapes = {}
        for p in self.p_set:
            tmp_md = self._model_class.Model(p).state_dict()
            self.pmodel_shapes[p] = {k:v.shape for k,v in tmp_md.items()}
        self.client_order = {}
        self.client_layer_start = {cid:{} for cid in range(self.num_clients)}

    def pack(self, client_id, mtype=0, *args, **kwargs):
        return {'w': self.generate_subdict(self.client_p[client_id], client_id)}

    def iterate(self):
        self.selected_clients = self.sample()
        models = self.communicate(self.selected_clients)['model']
        self.model = self.aggregate(models)
        for cid, c in enumerate(self.clients):
            c.model.load_state_dict(self.generate_subdict(c.p, cid, rolex_to_next=False))
        self.client_order = {}

    def generate_subdict(self, p, client_id = None, rolex_to_next=True):
        md = copy.deepcopy(self.model.state_dict())
        keys = list(md.keys())
        layer_order = {}
        order = None
        previous_order = None
        previous_num_out = None
        reslayer_order = None
        reslayer_num_out = None
        reslayer = None
        down_order = None
        layer_start = self.client_layer_start[client_id]
        for i in range(len(keys)):
            k = keys[i]
            if 'layer' in k:
                t1 = k.find('layer')
                t2 = k.find('.', t1)
                if reslayer != k[t1:t2]:
                    reslayer = k[t1:t2]
                    reslayer_order = previous_order
                    reslayer_num_out = previous_num_out
            num_out = md[k].shape[0]

            if 'downsample' in k:
                if '0.weight' in k:
                    layer_sid = layer_start.get(k, None)
                    if layer_sid is None:
                        self.client_layer_start[client_id][k] = layer_sid = 0
                    else:
                        layer_sid = (layer_sid+1)%num_out if rolex_to_next else layer_sid%num_out
                        self.client_layer_start[client_id][k] = layer_sid
                    down_order = sorted([(layer_sid+ni)%num_out for ni in range(math.ceil(num_out*p))])
                    s = list(md[k].shape)
                    v = (md[k].permute([1, 0, 2, 3]).reshape(reslayer_num_out, -1, s[0])[reslayer_order, :]).reshape(-1, s[0], 1, 1).permute([1, 0, 2, 3])
                    md[k] = v
                v = md[k][down_order]
                md[k] = v
                layer_order[k] = down_order
            # the number of output channels of the current layer
            elif 'weight' in k and 'bn' not in k:
                # select partial output channels
                layer_sid = layer_start.get(k, None)
                if layer_sid is None:
                    self.client_layer_start[client_id][k] = layer_sid = 0
                else:
                    layer_sid = (layer_sid+1)%num_out
                    self.client_layer_start[client_id][k] = layer_sid
                order = sorted([(layer_sid+ni)%num_out for ni in range(math.ceil(num_out*p))])
                if 'head.weight' in k:
                    order = torch.arange(md[k].shape[0]).tolist()
                # order = sorted(np.random.choice(list(range(num_out)), int(num_out*p), replace=False))
                # narrow the input channels of the next layer
                if previous_order is not None:
                    if previous_num_out != md[k].shape[1]: # linear after conv
                        s = list(md[k].shape)
                        v = (md[k].permute([1, 0]).reshape(previous_num_out, -1, s[0])[previous_order, :]).reshape(-1, s[
                            0]).permute([1, 0])
                        md[k] = v
                    else:
                        v = md[k][:, previous_order]
                        md[k] = v
            # narrow the out channels
                v = md[k][order]
                md[k] = v
                layer_order[k] = order
                previous_order = order
                previous_num_out = num_out
            else:
                v = md[k][order]
                md[k] = v
                layer_order[k] = order
                previous_order = order
                previous_num_out = num_out

        if client_id is not None: self.client_order[client_id] = layer_order
        return md

    def aggregate(self, models):
        mds = [mi.state_dict() for mi in models]
        full_model_shape = self.pmodel_shapes[1.0]
        sum_md = {k: torch.zeros(s) for k, s in full_model_shape.items()}
        sum_mask = {k: torch.zeros(s) for k, s in full_model_shape.items()}
        for idx, md in enumerate(mds):
            client_id = self.received_clients[idx]
            layer_order = self.client_order[client_id]
            keys = list(md.keys())
            previous_order = None
            reslayer_order = None
            reslayer = None
            for i in range(len(keys)):
                k = keys[i]
                if 'layer' in k:
                    t1 = k.find('layer')
                    t2 = k.find('.', t1)
                    if reslayer != k[t1:t2]:
                        reslayer = k[t1:t2]
                        reslayer_order = previous_order
                md[k] = md[k].cpu()
                if previous_order is None:
                    previous_order = torch.arange(md[k].shape[1])
                order = layer_order[k]
                if 'downsample.0.weight' in k:
                    tmp = torch.zeros_like(sum_md[k])
                    tmp_mask = torch.zeros_like(sum_md[k])
                    for j in range(len(order)):
                        tmp[order[j]][reslayer_order] += md[k][j]
                        tmp_mask[order[j]][reslayer_order] += 1
                    sum_md[k] += tmp
                    sum_mask[k] += tmp_mask
                elif 'weight' in k and 'bn' not in k and 'downsample.1' not in k:
                    if md[k].shape[1]!=len(previous_order): # linear after conv
                        num_items = int(md[k].shape[1]/len(previous_order))
                        new_order = []
                        for oi in previous_order:
                            new_order += [oi*num_items+x for x in range(num_items)]
                        previous_order = new_order
                    if torch.is_tensor(previous_order): previous_order = previous_order.tolist()
                    tmp = torch.zeros_like(sum_md[k])
                    tmp_mask = torch.zeros_like(sum_md[k])
                    for j in range(len(order)):
                        tmp[order[j]][previous_order] += md[k][j]
                        tmp_mask[order[j]][previous_order] += 1
                    previous_order = order
                    sum_md[k] += tmp
                    sum_mask[k] += tmp_mask
                else:
                    sum_md[k][order] += md[k]
                    sum_mask[k][order] += torch.ones_like(md[k])
        for k in sum_md.keys():
            sum_md[k] /= (sum_mask[k] + 1e-8)

        global_weights_last = self.model.state_dict()
        global_weights = copy.deepcopy(sum_md)
        step = 0
        for key in global_weights.keys():
            if 'weight' in key:
                global_weights[key] = torch.from_numpy(
                    np.where(sum_mask[key] < 1, global_weights_last[key].cpu(), sum_md[key].cpu()))
                step += 1
        self.model.load_state_dict(global_weights)
        # self.model.load_state_dict(sum_md)
        return self.model

class Client(fedavg.Client):
    def initialize(self, *args, **kwargs):
        self.model = self._model_class.Model(self.p).to(self.device)

    def unpack(self, received_pkg):
        w = received_pkg['w']
        self.model.load_state_dict(w)
        return self.model

def init_global_module(object):
    module_class = eval(os.path.split(object.option['task'])[-1].upper().split('_')[0]+'Model')
    return module_class.init_global_module(object)

def init_local_module(object):
    module_class = eval(os.path.split(object.option['task'])[-1].upper().split('_')[0]+'Model')
    return module_class.init_local_module(object)

def init_dataset(object):
    module_class = eval(os.path.split(object.option['task'])[-1].upper().split('_')[0]+'Model')
    return module_class.init_dataset(object)

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
            self.num_classes = 10
            self.encoder = usub.PResNetEncoder(usub.PBasicBlock, [2,2,2,2], norm_layer=lambda x: nn.GroupNorm(2, x), p=p)
            self.head = usub.PLinear(512, 10, p=p, keep_dim_out=True)

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
            self.conv1 = nn.Conv2d(3, int(pmax*64), kernel_size=11, stride=4, padding=2)
            self.conv2 = nn.Conv2d(int(pmax*64), int(pmax*192), kernel_size=5, padding=2)
            self.conv3 = nn.Conv2d(int(pmax*192), int(pmax*384), kernel_size=3, padding=1)
            self.conv4 = nn.Conv2d(int(pmax*384), int(pmax*256), kernel_size=3, padding=1)
            self.conv5 = nn.Conv2d(int(pmax*256), int(pmax*256), kernel_size=3, padding=1)
            self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
            self.fc1 = nn.Linear(int(pmax*256 * 6 * 6), int(pmax*1024))
            self.fc2 = nn.Linear(int(pmax*1024), int(pmax*1024))
            self.head = nn.Linear(int(pmax*1024), num_classes)

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
            x = self.encoder(x)
            x = self.head(x)
            return x

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
            self.conv1 = nn.Conv2d(3, int(64 * pmax), 5, bias=True)
            self.conv2 = nn.Conv2d(int(64 * pmax), int(64 * pmax), 5, bias=True)
            self.flatten = nn.Flatten(1)
            self.linear1 = nn.Linear(int(1600 * pmax), int(384 * pmax), bias=True)
            self.linear2 = nn.Linear(int(384 * pmax), int(192 * pmax), bias=True)
            self.head = nn.Linear(int(192 * pmax), 100, bias=True)


        def encoder(self, x):
            x = self.pool2d(self.relu(self.conv1(x)))
            x = self.pool2d(self.relu(self.conv2(x)))
            x = self.flatten(x)
            x = self.relu(self.linear1(x))
            x = self.relu(self.linear2(x))
            return x

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
