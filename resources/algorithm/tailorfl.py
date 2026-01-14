import copy
from collections import OrderedDict

import torch
import os
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
        self.init_algo_para({'lmbd':0.5, 'alpha':0.05, 'T1': 10, 'Lp':0.5, 'dp':0.0625})
        self.model = self._model_class.Model().to(self.device)
        self.client_filters = [None for _ in self.clients]
        self.client_correlations = torch.zeros(self.num_clients, self.num_clients)
        for c in self.clients:
            c.p = 1.0
            while c.p ** 2 > c._capacity:
                c.p -= self.dp
        self.client_ps = [c.p for c in self.clients]

    def pack(self, client_id, mtype=0, *args, **kwargs):
        if mtype==1:
            return {'model': copy.deepcopy(self.model)}
        else:
            client_filter = self.client_filters[client_id]
            md = self.generate_submodel(client_filter)
            return {'w': {k:v/self.client_ps[client_id] for k,v in md.items()}}

    def generate_submodel(self, filter):
        md = copy.deepcopy(self.model.state_dict())
        keys = list(filter.keys())
        res = {}
        # previous info
        previous_order = None
        previous_num_out = None
        reslayer_order = None
        reslayer_num_out = None
        reslayer = None
        for i in range(len(keys)):
            k = keys[i]
            # the number of output channels of the current layer
            num_out = md[k].shape[0]
            order = filter[k]
            # judge if layer changed
            if 'layer' in k:
                t1 = k.find('layer')
                t2 = k.find('.', t1)
                if reslayer != k[t1:t2]:
                    reslayer = k[t1:t2]
                    reslayer_order = previous_order
                    reslayer_num_out = previous_num_out
            # clip input channels
            if 'downsample.0.weight' in k:
                s = list(md[k].shape)
                v = (md[k].permute([1, 0, 2, 3]).reshape(reslayer_num_out, -1, s[0])[reslayer_order, :]).reshape(-1, s[0], 1, 1).permute([1, 0, 2, 3])
                md[k] = v
            elif previous_order is not None and 'weight' in k and 'bn' not in k and 'downsample' not in k:
                if previous_num_out != md[k].shape[1]: # linear after conv
                    s = list(md[k].shape)
                    v = (md[k].permute([1, 0]).reshape(previous_num_out, -1, s[0])[previous_order, :]).reshape(-1, s[0]).permute([1, 0])
                    md[k] = v
                else:
                    v = md[k][:, previous_order]
                    md[k] = v
            v = md[k][order]
            res[k] = v
            previous_order = order
            previous_num_out = num_out
        return res

    def iterate(self):
        if self.current_round==1:
            res = self.communicate([cid for cid in range(self.num_clients)], mtype=1)
            filters, thetas = res['filter'], res['theta']
            for i in range(self.num_clients):
                self.client_filters[i] = filters[i]
                self.client_correlations[i][i] = 1.0
                for j in range(i+1, self.num_clients):
                    dist = torch.cosine_similarity(thetas[i]-thetas[i].mean(), thetas[j]-thetas[j].mean(), dim=0)
                    self.client_correlations[i][j] = self.client_correlations[j][i] = dist
        else:
            self.selected_clients = self.sample()
            ws = self.communicate(self.selected_clients)['w']
            self.model = self.aggregate(ws)
            self.model_tuning()
            # set model for test
            self.communicate([cid for cid in range(self.num_clients)], mtype=2)
        return

    def aggregate(self, mds: list, *args, **kwargs):
        # scale submodels
        mds = [{k:v*self.client_ps[i] for k,v in md.items()} for i,md in zip(self.received_clients, mds)]
        sum_md = {k: torch.zeros_like(v).cpu() for k, v in self.model.state_dict().items() if 'imp' not in k}
        sum_mask = copy.deepcopy(sum_md)
        aggregation_weights = [len(self.clients[i].train_data)/self.client_ps[i] for i in range(len(self.received_clients))]
        for idx, md in enumerate(mds):
            client_id = self.received_clients[idx]
            layer_order = {k:v.cpu() for k,v in self.client_filters[client_id].items()}
            pre_order = None
            keys = list(md.keys())
            reslayer_order = None
            reslayer = None
            for i in range(len(keys)):
                k = keys[i]
                md[k] = md[k].cpu()
                if pre_order is None: pre_order = torch.arange(md[k].shape[1])
                order = layer_order[k]
                if 'layer' in k:
                    t1 = k.find('layer')
                    t2 = k.find('.', t1)
                    if reslayer != k[t1:t2]:
                        reslayer = k[t1:t2]
                        reslayer_order = pre_order
                # clip input channels
                if 'downsample.0.weight' in k:
                    if torch.is_tensor(reslayer_order): reslayer_order = reslayer_order.tolist()
                    shape = tuple(sum_md[k].shape)
                    mshape = tuple(sum_md[k].reshape(shape[0], -1).shape)
                    kernel_size = shape[-1]
                    tmp = np.zeros(mshape)
                    tmp_mask = np.zeros_like(tmp)
                    tmp_order = order.numpy()
                    tmp_pre_order = []
                    for j in reslayer_order:
                        tmp_pre_order += [kernel_size * kernel_size * j + ji for ji in range(kernel_size * kernel_size)]
                    tmp[np.ix_(tmp_order, tmp_pre_order)] = md[k].reshape(md[k].shape[0], -1).numpy()
                    tmp_mask[np.ix_(tmp_order, tmp_pre_order)] = 1
                    tmp = torch.from_numpy(tmp.reshape(shape))
                    tmp_mask = torch.from_numpy(tmp_mask.reshape(shape))
                    sum_md[k] += tmp * aggregation_weights[idx]
                    sum_mask[k] += tmp_mask * aggregation_weights[idx]
                elif 'weight' in k and 'bn' not in k and 'downsample' not in k:
                    if md[k].shape[1]!=len(pre_order): # linear after conv
                        num_items = int(md[k].shape[1]/len(pre_order))
                        new_order = []
                        for oi in pre_order:
                            new_order += [oi*num_items+x for x in range(num_items)]
                        pre_order = new_order
                    if torch.is_tensor(pre_order): pre_order = pre_order.tolist()
                    # tmp = torch.zeros_like(sum_md[k])
                    # tmp_mask = torch.zeros_like(sum_md[k])
                    # for j in range(len(order)):
                    #     tmp[order[j]][pre_order] += md[k][j]
                    #     tmp_mask[order[j]][pre_order] += 1
                    shape = tuple(sum_md[k].shape)
                    if len(shape)==4:
                        mshape = tuple(sum_md[k].reshape(shape[0], -1).shape)
                        kernel_size = shape[-1]
                        tmp = np.zeros(mshape)
                        tmp_mask = np.zeros_like(tmp)
                        tmp_order = order.numpy()
                        tmp_pre_order = []
                        for j in pre_order:
                            tmp_pre_order += [kernel_size*kernel_size*j+ji for ji in range(kernel_size*kernel_size)]
                        tmp[np.ix_(tmp_order, tmp_pre_order)] = md[k].reshape(md[k].shape[0], -1).numpy()
                        tmp_mask[np.ix_(tmp_order, tmp_pre_order)] = 1
                        tmp = torch.from_numpy(tmp.reshape(shape))
                        tmp_mask = torch.from_numpy(tmp_mask.reshape(shape))
                    else:
                        tmp = np.zeros(shape)
                        tmp_mask = np.zeros_like(tmp)
                        tmp_order = order.numpy()
                        tmp_pre_order = pre_order
                        tmp[np.ix_(tmp_order, tmp_pre_order)] = md[k].reshape(md[k].shape[0], -1).numpy()
                        tmp_mask[np.ix_(tmp_order, tmp_pre_order)] = 1
                        tmp = torch.from_numpy(tmp)
                        tmp_mask = torch.from_numpy(tmp_mask)
                    pre_order = order
                    # scale the submodel according to the prune rate
                    sum_md[k] += tmp * aggregation_weights[idx]
                    sum_mask[k] += tmp_mask * aggregation_weights[idx]
                else:
                    sum_md[k][order] += md[k] * aggregation_weights[idx]
                    sum_mask[k][order] += torch.ones_like(md[k]) * aggregation_weights[idx]
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
        global_weights_last.update(global_weights)
        self.model.load_state_dict(global_weights_last)
        return self.model

    def model_tuning(self):
        md = self.model.state_dict()
        keys = list(md.keys())
        all_num_layers = 0
        reslayer = None
        for k in keys:
            if 'layer' in k:
                t1 = k.find('layer')
                t2 = k.find('.', t1)
                if reslayer != k[t1:t2]:
                    reslayer = k[t1:t2]
                    all_num_layers += 1
            elif 'bn' in k:
                continue
            elif 'weight' in k:
                all_num_layers += 1
        num_shared_layers = int(all_num_layers * (1 - self.Lp))
        lid = -1
        res = {}
        for kid in range(len(keys)):
            k = keys[kid]
            if 'layer' in k:
                t1 = k.find('layer')
                t2 = k.find('.', t1)
                if reslayer != k[t1:t2]:
                    reslayer = k[t1:t2]
                    lid += 1
            elif 'bn' in k or 'imp' in k:
                continue
            elif 'weight' in k:
                lid += 1
            if 'imp' in k or 'bias' in k or 'bn' in k or 'downsample.2' in k: continue
            if lid < num_shared_layers:
                weight_magnitude = md[k] ** 2
                while len(weight_magnitude.shape) > 1:
                    weight_magnitude = weight_magnitude.sum(dim=-1)
                bias_key = keys[kid + 1] if kid + 1 < len(keys) else None
                if bias_key is not None and bias_key == k.replace('weight', 'bias'):
                    weight_magnitude += md[bias_key] ** 2
                selected_channels = torch.argsort(weight_magnitude, descending=True)
                for cid in range(self.num_clients):
                    self.client_filters[cid][k] = selected_channels[:int(md[k].shape[0]*self.client_ps[cid])]
                res[k] = selected_channels
                if bias_key is not None and bias_key == k.replace('weight', 'bias'):
                    res[bias_key] = selected_channels
                    for cid in range(self.num_clients):
                        self.client_filters[cid][bias_key] = selected_channels[:int(md[bias_key].shape[0] * self.client_ps[cid])]
                    bn_idx = kid+2
                else:
                    bn_idx = kid+1
                if bn_idx<len(keys) and 'imp' in keys[bn_idx]: bn_idx+=1
                while bn_idx<len(keys) and ('bn' in keys[bn_idx] or 'downsample.2' in keys[bn_idx]):
                    for cid in range(self.num_clients):
                        self.client_filters[cid][keys[bn_idx]] = selected_channels[:int(md[keys[bn_idx]].shape[0] * self.client_ps[cid])]
                    bn_idx += 1
                    if 'conv' in keys[bn_idx]: break
            else:
                all_filters = [copy.deepcopy(self.client_filters[i][k]) for i in range(self.num_clients)]
                for ci in range(self.num_clients):
                    si = self.client_correlations[ci]
                    crt_filter = all_filters[ci].cpu()
                    element_count = 0
                    crt_set = set(crt_filter.tolist())
                    if len(crt_filter)==md[k].shape[0]:continue
                    while True:
                        target_i = [si[cj] * min(len(crt_filter), len(self.client_filters[cj][k])) for cj in range(self.num_clients)]
                        curr_i = [len(crt_set.intersection(set(self.client_filters[cj][k].tolist()))) for cj in range(self.num_clients)]
                        delta_i = [ti-ci for ti,ci in zip(target_i, curr_i)]
                        filter_weights = torch.tensor([0.0 for _ in range(md[k].shape[0])])
                        for cj, dij in enumerate(delta_i):
                            if cj==ci:continue
                            filter_weights[self.client_filters[cj][k].cpu()] += dij
                        out_filter = torch.tensor([x for x in range(md[k].shape[0]) if x not in crt_set], dtype=torch.int64)
                        minf_idx = torch.argmin(filter_weights[crt_filter])
                        minfw = filter_weights[crt_filter][minf_idx]
                        minf = int(crt_filter[int(minf_idx)])
                        maxf_idx = torch.argmax(filter_weights[out_filter])
                        maxfw = filter_weights[out_filter][maxf_idx]
                        maxf = int(out_filter[int(maxf_idx)])
                        if maxfw>minfw:
                            crt_set.remove(minf)
                            crt_set.add(maxf)
                            crt_filter = torch.tensor(sorted(list(crt_set)), dtype=torch.int64)
                            element_count += 1
                            if element_count>= int(self.alpha*min(len(crt_filter), md[k].shape[0]-len(crt_filter))):
                                break
                        else:
                            break
                    all_filters[ci] = crt_filter
                for ci in range(self.num_clients):
                    self.client_filters[ci][k] = all_filters[ci]

                j = kid + 1
                while j < len(keys) and 'imp' not in keys[j]: j += 1
                if j == len(keys): continue
                if kid+1<len(keys) and k.replace('weight', 'bias') == keys[kid + 1]:
                    for ci in range(self.num_clients):
                        self.client_filters[ci][keys[kid + 1]] = all_filters[ci]
                    bn_idx = kid +2
                else:
                    bn_idx = kid + 1
                if bn_idx<len(keys) and 'imp' in keys[bn_idx]: bn_idx+=1
                while bn_idx<len(keys) and ('bn' in keys[bn_idx] or 'downsample.2' in keys[bn_idx]):
                    for ci in range(self.num_clients):
                        self.client_filters[ci][keys[bn_idx]] = all_filters[ci]
                    bn_idx += 1
                    if 'conv' in keys[bn_idx]: break
            lid += 1
        return

class Client(fedavg.Client):
    def initialize(self, *args, **kwargs):
        self.actions = {0:self.reply, 1:self.prune_model, 2:self.set_test_model}
        self.model = self._model_class.Model(self.p)
        for n,p in self.model.named_parameters():
            if 'imp' in n:
                p.requires_grad = False

    def prune_model(self, svr_pkg):
        model = svr_pkg['model']
        for n,p in model.named_parameters():
            if 'imp' not in n:
                p.requires_grad = False
        optimizer = torch.optim.SGD([{'params': [p for p in model.parameters() if p.requires_grad]}], lr=self.learning_rate, momentum=self.momentum)
        train_loader = self.calculator.get_dataloader(self.train_data, self.batch_size)
        i=0
        while True:
            if i>=self.T1:
                break
            for _, batch_data in enumerate(train_loader):
                batch_data = self.calculator.to_device(batch_data)
                output = model(batch_data[0], use_importance=True)
                loss = self.calculator.criterion(output, batch_data[-1])
                loss_reg = 0.0
                num_paras = 0
                for n,p in model.named_parameters():
                    if 'imp' in n:
                        loss_reg += (p**2).sum()
                        num_paras += p.numel()
                if loss_reg>0.: loss += self.lmbd*loss_reg/num_paras
                loss.backward()
                optimizer.step()
                with torch.no_grad():
                    for n, p in model.named_parameters():
                        if 'imp' in n:
                            p.data = torch.clamp(p.data, 0, 1)
                i += 1
                if i>=self.T1:
                    break

        # prune model
        md = model.state_dict()
        reslayer = None
        keys = list(md.keys())
        all_num_layers = 0
        for k in keys:
            if 'layer' in k:
                t1 = k.find('layer')
                t2 = k.find('.', t1)
                if reslayer != k[t1:t2]:
                    reslayer = k[t1:t2]
                    all_num_layers += 1
            elif 'bn' in k:
                continue
            elif 'weight' in k:
                all_num_layers += 1
        num_shared_layers = int(all_num_layers * (1 - self.Lp))
        # num_shared_layers = int(len([k for k in md if 'weight' in k]) * (1 - self.Lp))
        l = -1
        res = {}
        reslayer = None
        for i in range(len(keys)):
            k = keys[i]
            if 'layer' in k:
                t1 = k.find('layer')
                t2 = k.find('.', t1)
                if reslayer != k[t1:t2]:
                    reslayer = k[t1:t2]
                    l += 1
            elif 'bn' in k or 'imp' in k:
                continue
            elif 'weight' in k:
                l += 1
            if 'imp' in k or 'bn' in k or 'bias' in k or 'downsample.2' in k: continue
            if l<num_shared_layers:
                weight_magnitude = md[k]**2
                while len(weight_magnitude.shape)>1:
                    weight_magnitude = weight_magnitude.sum(dim=-1)
                if i+1<len(keys) and k.replace('weight', 'bias')==keys[i+1]:
                    vbias = md[keys[i+1]]
                    weight_magnitude += vbias**2
                else:
                    vbias = None
                selected_channels = torch.sort(torch.argsort(weight_magnitude, descending=True)[:int(md[k].shape[0]*self.p)]).values
                res[k] = selected_channels
                if vbias is not None:
                    res[keys[i+1]] = selected_channels
                    bn_idx = i+2
                else:
                    bn_idx = i+1
                if 'imp' in keys[bn_idx]: bn_idx+=1
                while bn_idx<len(keys) and ('bn' in keys[bn_idx] or 'downsample.2' in keys[bn_idx]):
                    res[keys[bn_idx]] = selected_channels
                    bn_idx += 1
                    if 'conv' in keys[bn_idx]: break
            else:
                j = i+1
                while j<len(keys) and 'imp' not in keys[j]: j+=1
                if j==len(keys):
                    if 'head' not in k:
                        continue
                    else:
                        res[k] = torch.arange(md[k].shape[0])
                        if i + 1 < len(keys) and k.replace('weight', 'bias') == keys[i + 1]:
                            res[keys[i + 1]] = res[k]
                        break
                importance = md[keys[j]]**2
                selected_channels = torch.sort(torch.argsort(importance, descending=True)[:int(md[k].shape[0]*self.p)]).values
                res[k] = selected_channels
                if i+1<len(keys) and k.replace('weight', 'bias')==keys[i+1]:
                    res[keys[i+1]] = selected_channels
                    bn_idx = i + 2
                else:
                    if i+1<len(keys):
                        bn_idx = i+1
                    else:
                        continue
                if 'imp' in keys[bn_idx]: bn_idx+=1
                while bn_idx < len(keys) and ('bn' in keys[bn_idx] or 'downsample.2' in keys[bn_idx]):
                    res[keys[bn_idx]] = selected_channels
                    bn_idx += 1
                    if 'conv' in keys[bn_idx]: break


        last_theta = None
        for i in range(len(keys)):
            if 'imp' in keys[len(keys)-1-i]:
                last_theta = md[keys[len(keys)-1-i]]
                break
        # init local model
        return {'filter': res, 'theta': last_theta}

    def set_test_model(self, received_pkg):
        md = self.model.state_dict()
        md.update(received_pkg['w'])
        self.model.load_state_dict(md)

    def unpack(self, received_pkg):
        md = self.model.state_dict()
        md.update(received_pkg['w'])
        self.model.load_state_dict(md)
        train_model = copy.deepcopy(self.model)
        return train_model

    def pack(self, model, *args, **kwargs):
        return {'w': {k:v for k,v in model.state_dict().items() if 'imp' not in k}}

class ImportanceLayer(nn.Module):
    def __init__(self, num_channels: int):
        super().__init__()
        self.num_channels = num_channels
        self.importance = nn.Parameter(torch.ones(num_channels))

    def forward(self, input, use_importance=False):
        s = input.shape
        return (input.view(s[0], s[1], -1) * (torch.repeat_interleave(self.importance.unsqueeze(0), s[0], dim=0).unsqueeze(dim=2))).view(s) if use_importance else input

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

    class TPBasicBlock(nn.Module):
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
                p:float = 1.0
        ) -> None:
            super().__init__()
            if norm_layer is None:
                norm_layer = nn.BatchNorm2d
            if groups != 1 or base_width != 64:
                raise ValueError("BasicBlock only supports groups=1 and base_width=64")
            if dilation > 1:
                raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
            # Both self.conv1 and self.downsample layers downsample the input when stride != 1
            self.conv1 = usub.PConv2d(inplanes, planes, 3, stride=stride, padding=dilation, p=p)
            self.imp1 = ImportanceLayer(int(p * planes))
            self.bn1 = norm_layer(int(p*planes))
            self.relu = nn.ReLU(inplace=True)
            self.conv2 = usub.PConv2d(planes, planes,3, padding=1, p=p)
            self.imp2 = ImportanceLayer(int(p * planes))
            self.bn2 = norm_layer(int(p*planes))
            self.downsample = downsample
            self.stride = stride

        def forward(self, x, use_importance=False):
            identity = x

            out = self.conv1(x)
            out = self.imp1(out, use_importance)
            out = self.bn1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.imp2(out, use_importance)
            out = self.bn2(out)

            if self.downsample is not None:
                identity = self.downsample(x)

            out += identity
            out = self.relu(out)

            return out

    class TPResNetEncoder(nn.Module):
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
            self.base_width = width_per_group
            self.conv1 = usub.PConv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False, keep_dim_in=True,
                                 p=p)
            self.imp0 = ImportanceLayer(int(p*self.inplanes))
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
                    if isinstance(m, CIFAR10Model.TPBasicBlock) and m.bn2.weight is not None:
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
                    ImportanceLayer(int(planes * block.expansion * self.p)),
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

        def _forward_impl(self, x, use_importance=False):
            # See note [TorchScript super()]
            x = self.conv1(x)
            x = self.imp0(x, use_importance)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)
            for i in range(4):
                layer = getattr(self, f'layer{i + 1}')
                for li in layer:
                    x = li(x, use_importance)

            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            return x

        def forward(self, x, use_importance=False):
            return self._forward_impl(x, use_importance)

    class Model(fuf.FModule):
        def __init__(self, p: float = 1.0):
            super().__init__()
            self.num_classes = 10
            self.encoder = CIFAR10Model.TPResNetEncoder(CIFAR10Model.TPBasicBlock, [2,2,2,2], norm_layer=lambda x: nn.GroupNorm(2, x), p=p)
            self.head = usub.PLinear(512*CIFAR10Model.TPBasicBlock.expansion, 10, p=p, keep_dim_out=True)

        def forward(self, x, use_importance=False):
            x = self.encoder(x, use_importance)
            return self.head(x)

    # class Model(fuf.FModule):
    #     def __init__(self, rate: float = 1.0):
    #         super().__init__()
    #         self.num_classes = 10
    #         self.relu = nn.ReLU()
    #         self.maxpool = nn.MaxPool2d(2)
    #         self.conv1 =  nn.Conv2d(3, int(64*rate), 5, bias=True)
    #         self.imp1 = ImportanceLayer(int(64*rate))
    #         self.conv2 = nn.Conv2d(int(64*rate), int(64*rate), 5, bias=True)
    #         self.imp2 = ImportanceLayer(int(64*rate))
    #         self.flatten = nn.Flatten(1)
    #         self.fc1 = nn.Linear(int(1600 * rate), int(384 * rate), bias=True)
    #         self.imp3 = ImportanceLayer(int(384 * rate))
    #         self.fc2 = nn.Linear(int(384*rate), int(192*rate), bias=True)
    #         self.imp4 = ImportanceLayer(int(192 * rate))
    #         self.head = nn.Linear(int(192*rate), 10, bias=True)
    #         self.imps = [getattr(self, 'imp'+str(i)) for i in range(1, 5)]
    #
    #     def encoder(self, x, use_importance=False):
    #         x = self.relu(self.conv1(x))
    #         x = self.imp1(x, use_importance)
    #         x = self.maxpool(x)
    #         x = self.relu(self.conv2(x))
    #         x = self.imp2(x, use_importance)
    #         x = self.maxpool(x)
    #         x = self.flatten(x)
    #         x = self.relu(self.fc1(x))
    #         x = self.imp3(x, use_importance)
    #         x = self.relu(self.fc2(x))
    #         x = self.imp4(x, use_importance)
    #         return x
    #
    #     def forward(self, x, use_importance=False):
    #         x = self.encoder(x, use_importance)
    #         return self.head(x)

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
                return cls.Model(object.rate ** object.p)

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
        def __init__(self, rate: float = 1.0):
            super().__init__()
            self.num_classes = 100
            self.relu = nn.ReLU()
            self.maxpool = nn.MaxPool2d(2)
            self.conv1 =  nn.Conv2d(3, int(64*rate), 5, bias=True)
            self.imp1 = ImportanceLayer(int(64*rate))
            self.conv2 = nn.Conv2d(int(64*rate), int(64*rate), 5, bias=True)
            self.imp2 = ImportanceLayer(int(64*rate))
            self.flatten = nn.Flatten(1)
            self.fc1 = nn.Linear(int(1600 * rate), int(384 * rate), bias=True)
            self.imp3 = ImportanceLayer(int(384 * rate))
            self.fc2 = nn.Linear(int(384*rate), int(192*rate), bias=True)
            self.imp4 = ImportanceLayer(int(192 * rate))
            self.head = nn.Linear(int(192*rate), 100, bias=True)
            self.imps = [getattr(self, 'imp'+str(i)) for i in range(1, 5)]

        def encoder(self, x, use_importance=False):
            x = self.relu(self.conv1(x))
            x = self.imp1(x, use_importance)
            x = self.maxpool(x)
            x = self.relu(self.conv2(x))
            x = self.imp2(x, use_importance)
            x = self.maxpool(x)
            x = self.flatten(x)
            x = self.relu(self.fc1(x))
            x = self.imp3(x, use_importance)
            x = self.relu(self.fc2(x))
            x = self.imp4(x, use_importance)
            return x

        def forward(self, x, use_importance=False):
            x = self.encoder(x, use_importance)
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
                return cls.Model(object.rate ** object.p)

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
        def __init__(self, rate: float = 1.0):
            super().__init__()
            self.num_classes = 10
            self.relu = nn.ReLU()
            self.maxpool = nn.MaxPool2d(2)
            self.conv1 =  nn.Conv2d(1, int(32*rate), 5, bias=True, padding=2)
            self.imp1 = ImportanceLayer(int(32*rate))
            self.conv2 = nn.Conv2d(int(32*rate), int(64*rate), 5, bias=True, padding=2)
            self.imp2 = ImportanceLayer(int(64*rate))
            self.flatten = nn.Flatten(1)
            self.fc1 = nn.Linear(int(3136 * rate), int(512* rate), bias=True)
            self.imp3 = ImportanceLayer(int(512 * rate))
            self.fc2 = nn.Linear(int(512*rate), int(128*rate), bias=True)
            self.imp4 = ImportanceLayer(int(128 * rate))
            self.head = nn.Linear(int(128*rate), 10, bias=True)
            self.imps = [getattr(self, 'imp'+str(i)) for i in range(1, 5)]

        def encoder(self, x, use_importance=False):
            x = self.relu(self.conv1(x))
            x = self.imp1(x, use_importance)
            x = self.maxpool(x)
            x = self.relu(self.conv2(x))
            x = self.imp2(x, use_importance)
            x = self.maxpool(x)
            x = self.flatten(x)
            x = self.relu(self.fc1(x))
            x = self.imp3(x, use_importance)
            x = self.relu(self.fc2(x))
            x = self.imp4(x, use_importance)
            return x

        def forward(self, x, use_importance=False):
            x = self.encoder(x, use_importance)
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
                return cls.Model(object.rate ** object.p)

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
        def __init__(self, rate: float = 1.0):
            super().__init__()
            self.num_classes = 10
            self.relu = nn.ReLU()
            self.maxpool = nn.MaxPool2d(2)
            self.conv1 =  nn.Conv2d(1, int(32*rate), 5, bias=True, padding=2)
            self.imp1 = ImportanceLayer(int(32*rate))
            self.conv2 = nn.Conv2d(int(32*rate), int(64*rate), 5, bias=True, padding=2)
            self.imp2 = ImportanceLayer(int(64*rate))
            self.flatten = nn.Flatten(1)
            self.fc1 = nn.Linear(int(3136 * rate), int(512* rate), bias=True)
            self.imp3 = ImportanceLayer(int(512 * rate))
            self.fc2 = nn.Linear(int(512*rate), int(128*rate), bias=True)
            self.imp4 = ImportanceLayer(int(128 * rate))
            self.head = nn.Linear(int(128*rate), 10, bias=True)
            self.imps = [getattr(self, 'imp'+str(i)) for i in range(1, 5)]

        def encoder(self, x, use_importance=False):
            x = self.relu(self.conv1(x))
            x = self.imp1(x, use_importance)
            x = self.maxpool(x)
            x = self.relu(self.conv2(x))
            x = self.imp2(x, use_importance)
            x = self.maxpool(x)
            x = self.flatten(x)
            x = self.relu(self.fc1(x))
            x = self.imp3(x, use_importance)
            x = self.relu(self.fc2(x))
            x = self.imp4(x, use_importance)
            return x

        def forward(self, x, use_importance=False):
            x = self.encoder(x, use_importance)
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
                return cls.Model(object.rate ** object.p)

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
        """
        used for DomainNet and Office-Caltech10
        """

        def __init__(self, rate:float=1.0, num_classes=10):
            super().__init__()
            self.pmax = rate
            self.relu = nn.ReLU(inplace=True)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2)
            self.conv1 = nn.Conv2d(3, int(rate * 64), kernel_size=11, stride=4, padding=2)
            self.imp1 = ImportanceLayer(int(rate * 64))
            self.conv2 = nn.Conv2d(int(rate * 64), int(rate * 192), kernel_size=5, padding=2)
            self.imp2 = ImportanceLayer(int(rate * 192))
            self.conv3 = nn.Conv2d(int(rate * 192), int(rate * 384), kernel_size=3, padding=1)
            self.imp3 = ImportanceLayer(int(rate * 384))
            self.conv4 = nn.Conv2d(int(rate * 384), int(rate * 256), kernel_size=3, padding=1)
            self.imp4 = ImportanceLayer(int(rate * 256))
            self.conv5 = nn.Conv2d(int(rate * 256), int(rate * 256), kernel_size=3, padding=1)
            self.imp5 = ImportanceLayer(int(rate * 256))
            self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
            self.fc1 = nn.Linear(int(rate * 256 * 6 * 6), int(rate * 4096))
            self.imp6 = ImportanceLayer(int(rate * 4096))
            self.fc2 = nn.Linear(int(rate * 4096), int(rate * 4096))
            self.imp7 = ImportanceLayer(int(rate * 4096))
            self.head = nn.Linear(int(rate * 4096), num_classes)

        def encoder(self, x, use_importance=False):
            x = self.relu(self.conv1(x))
            x = self.imp1(x, use_importance)
            x = self.maxpool(x)
            x = self.relu(self.conv2(x))
            x = self.imp2(x, use_importance)
            x = self.maxpool(x)
            x = self.relu(self.conv3(x))
            x = self.imp3(x, use_importance)
            x = self.relu(self.conv4(x))
            x = self.imp4(x, use_importance)
            x = self.relu(self.conv5(x))
            x = self.imp5(x, use_importance)
            x = self.maxpool(x)
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.relu(self.fc1(x))
            x = self.imp6(x, use_importance)
            x = self.relu(self.fc2(x))
            x = self.imp7(x, use_importance)
            return x

        def forward(self, x, use_importance=False):
            x = self.encoder(x, use_importance)
            x = self.head(x)
            return x

def init_global_module(object):
    module_class = eval(os.path.split(object.option['task'])[-1].upper().split('_')[0]+'Model')
    return module_class.init_global_module(object)

def init_local_module(object):
    module_class = eval(os.path.split(object.option['task'])[-1].upper().split('_')[0]+'Model')
    return module_class.init_local_module(object)

def init_dataset(object):
    module_class = eval(os.path.split(object.option['task'])[-1].upper().split('_')[0]+'Model')
    return module_class.init_dataset(object)