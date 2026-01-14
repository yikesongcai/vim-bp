import copy
from collections import OrderedDict

import torch
import os
import numpy as np
import flgo.algorithm.fedavg as fedavg
import flgo.utils.fmodule as fuf
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision.transforms import RandomCrop, RandomHorizontalFlip
import torchvision

class CIFAR10Model:
    class Model(fuf.FModule):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Sequential(
            OrderedDict([
                ('conv1', nn.Conv2d(3, 64, 5)),
                ('relu1', nn.ReLU(inplace=True)),
                ('maxpool1', nn.MaxPool2d(2)),

                ('conv2', nn.Conv2d(64, 64, 5)),
                ('relu2', nn.ReLU(inplace=True)),
                ('maxpool2', nn.MaxPool2d(2)),
                ('flatten', nn.Flatten(1)),

                ('linear1', nn.Linear(1600, 384)),
                ('relu3', nn.ReLU(inplace=True)),

                ('linear2', nn.Linear(384, 192)),
                ('relu4', nn.ReLU(inplace=True)),
            ])
            )
            self.head = nn.Linear(192, 10)

        def forward(self, x):
            x = self.encoder(x)
            return self.head(x)

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

    @classmethod
    def init_dataset(cls, object):
        if 'Client' in object.get_classname():
            object.train_data = cls.AugmentDataset(object.train_data)

    @classmethod
    def init_local_module(cls, object):
        pass

    @classmethod
    def init_global_module(cls, object):
        if 'Server' in object.__class__.__name__:
            object.model = cls.Model().to(object.device)

class DOMAINNETModel:
    class Model(fuf.FModule):
        """
        used for DomainNet and Office-Caltech10
        """

        def __init__(self, num_classes=10):
            super().__init__()
            self.features = nn.Sequential(
                OrderedDict([
                    ('conv1', nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2)),
                    ('relu1', nn.ReLU(inplace=True)),
                    ('maxpool1', nn.MaxPool2d(kernel_size=3, stride=2)),

                    ('conv2', nn.Conv2d(64, 192, kernel_size=5, padding=2)),
                    ('relu2', nn.ReLU(inplace=True)),
                    ('maxpool2', nn.MaxPool2d(kernel_size=3, stride=2)),

                    ('conv3', nn.Conv2d(192, 384, kernel_size=3, padding=1)),
                    ('relu3', nn.ReLU(inplace=True)),

                    ('conv4', nn.Conv2d(384, 256, kernel_size=3, padding=1)),
                    ('relu4', nn.ReLU(inplace=True)),

                    ('conv5', nn.Conv2d(256, 256, kernel_size=3, padding=1)),
                    ('relu5', nn.ReLU(inplace=True)),
                    ('maxpool5', nn.MaxPool2d(kernel_size=3, stride=2)),
                ])
            )
            self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
            self.fc1 = nn.Linear(256 * 6 * 6, 4096)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(4096, 4096)
            self.head = nn.Linear(4096, num_classes)

        def encoder(self, x):
            x = self.features(x)
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc1(x)
            x = self.relu(x)
            x = self.fc2(x)
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
        pass

    @classmethod
    def init_global_module(cls, object):
        if 'Server' in object.__class__.__name__:
            object.model = cls.Model().to(object.device)

class Server(fedavg.Server):
    def initialize(self, *args, **kwargs):
        self.init_algo_para({'lmbd':1e-4, 'prune_start_acc':0.2, 'prune_rate':0.1, 'mask_ratio':0.5})
        self.init_weight = copy.deepcopy(self.model.state_dict())

    def pack(self, client_id, mtype=0, *args, **kwargs):
        return {'model': copy.deepcopy(self.model)}

    def iterate(self):
        self.selected_clients = self.sample()
        res = self.communicate(self.selected_clients)
        masks, models = res['mask'], res['model']
        self.model = self.aggregate(models, masks)
        for c in self.clients:
            c.model = c.mask_model(copy.deepcopy(self.model), c.mask)
        # for cid in self.selected_clients: self.client_has_been_selected[cid] = True
        return

    def aggregate(self, models: list, masks: list):
        w = [mi.state_dict() for mi in models]
        w_avg = copy.deepcopy(w[0])
        w_avg = {k:v.cpu() for k, v in w_avg.items()}
        mask = [np.zeros_like(masks[0][j]) for j in range(len(masks[0]))]
        for i in range(len(masks)):
            for j in range(len(mask)):
                mask[j]+= masks[i][j]
        step = 0
        for key in w_avg.keys():
            if 'weight' in key:
                for i in range(1, len(w)):
                    w_avg[key] += w[i][key].cpu()*masks[i][step]
                w_avg[key] = torch.from_numpy(np.where(mask[step] < 1, 0, w_avg[key].cpu().numpy() / mask[step]))
                step += 1
            else:
                for i in range(1, len(w)):
                    w_avg[key] += w[i][key].cpu()
                w_avg[key] = torch.div(w_avg[key], len(w))
        global_weights_last = self.model.state_dict()
        global_weights = copy.deepcopy(w_avg)
        step = 0
        for key in global_weights.keys():
            if 'weight' in key:
                global_weights[key] = torch.from_numpy(
                    np.where(mask[step] < 1, global_weights_last[key].cpu(), w_avg[key].cpu()))
                step += 1
        self.model.load_state_dict(global_weights)
        return self.model

class Client(fedavg.Client):
    def initialize(self, *args, **kwargs):
        self.current_prune_rate = 1.0
        self.init_weight = None
        self.model = None
        self.mask = self.init_mask(self.server.model)
        self.prune_end_rate = self._capacity

    def unpack(self, received_pkg):
        train_model = received_pkg['model']
        self.mask_model(train_model, self.mask)
        acc_before_train = self.test(train_model, 'val')['accuracy']
        if acc_before_train>self.prune_start_acc and self.current_prune_rate>self.prune_end_rate:
            # mask global model from initial weights when the accuracy is adeduately high and the prune rate is not adeduately low (i.e. the number of parameters is too large)
            # once the accuracy is low or the prune rate achieves the target rate, stop clipping
            self.prune_by_percentile(train_model, self.mask)
            self.current_prune_rate = self.current_prune_rate * (1 - self.prune_rate)
            self.mask_model(train_model, self.mask)
        return train_model

    def pack(self, model, *args, **kwargs):
        self.mask_model(model, self.mask, model.state_dict())
        self.model = model
        return {'mask': self.mask, 'model': self.model}

    def sparse_reg(self, model):
        loss = 0.0
        for n,p in model.named_parameters():
            if 'weight' in n:
                ps = (p ** 2)
                if 'conv' in n:
                    # channel loss
                    loss += torch.sqrt(ps.sum(dim=-1).sum(dim=-1).sum(dim=-1)).sum()
                    # filter loss
                    loss += torch.sqrt(ps.sum(dim=-1).sum(dim=-1)).sum()
                elif 'linear' in n or 'head' in n or 'fc' in n:
                    # column and row loss
                    loss += (torch.sqrt(ps.sum(dim=0)).sum() + torch.sqrt(ps.sum(dim=1)).sum())
        return loss

    @fuf.with_multi_gpus
    def train(self, model):
        model.train()
        optimizer = self.calculator.get_optimizer(model, lr=self.learning_rate, weight_decay=self.weight_decay, momentum=self.momentum)
        for iter in range(self.num_steps):
            # get a batch of data
            batch_data = self.get_batch_data()
            model.zero_grad()
            # calculate the loss of the model on batched dataset through task-specified calculator
            loss = self.calculator.compute_loss(model, batch_data)['loss']
            loss += self.lmbd*self.sparse_reg(model)
            loss.backward()
            if self.clip_grad>0:torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=self.clip_grad)
            for name, p in model.named_parameters():
                if 'weight' in name:
                    tensor = p.data.cpu().numpy()
                    grad_tensor = p.grad.data.cpu().numpy()
                    grad_tensor = np.where(abs(tensor) < 1e-6, 0, grad_tensor)
                    p.grad.data = torch.from_numpy(grad_tensor).to(self.device)
            optimizer.step()
        return

    def prune_by_percentile(self, model, mask):
        # Calculate percentile value
        step = 0
        for name, param in model.named_parameters():
            # We do not prune bias term
            if 'weight' in name:
                tensor = param.data.cpu().numpy()
                alive = tensor[np.nonzero(tensor)] # flattened array of nonzero values
                percentile_value = np.percentile(abs(alive), self.prune_rate * 100)

                # Convert Tensors to numpy and calculate
                weight_dev = param.device
                new_mask = np.where(abs(tensor) < percentile_value, 0, mask[step])

                # Apply new weight and mask
                param.data = torch.from_numpy(tensor * new_mask).to(weight_dev)
                mask[step] = new_mask
                step += 1
        return

    def mask_model(self, model, mask, state_dict=None):
        step = 0
        if state_dict is None: state_dict = model.state_dict()
        for name, param in model.named_parameters():
            if "weight" in name:
                weight_dev = param.device
                param.data = torch.from_numpy(mask[step] * state_dict[name].cpu().numpy()).to(weight_dev)
                step = step + 1
            if "bias" in name:
                param.data = state_dict[name]
        return model

    def init_mask(self, model):
        return [np.ones_like(param.data.cpu().numpy()) for name, param in model.named_parameters() if 'weight' in name]

def init_global_module(object):
    module_class = eval(os.path.split(object.option['task'])[-1].upper().split('_')[0]+'Model')
    return module_class.init_global_module(object)

def init_local_module(object):
    module_class = eval(os.path.split(object.option['task'])[-1].upper().split('_')[0]+'Model')
    return module_class.init_local_module(object)

def init_dataset(object):
    module_class = eval(os.path.split(object.option['task'])[-1].upper().split('_')[0]+'Model')
    return module_class.init_dataset(object)