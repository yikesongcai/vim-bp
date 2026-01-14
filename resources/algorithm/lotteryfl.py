import copy
import torch
import numpy as np
import flgo.algorithm.fedavg as fedavg
import flgo.utils.fmodule as fuf

class Server(fedavg.Server):
    def initialize(self, *args, **kwargs):
        self.init_algo_para({'prune_start_acc':0.2, 'prune_end_rate':0.5, 'prune_rate':0.1, 'mask_ratio':0.5})
        self.init_weight = copy.deepcopy(self.model.state_dict())
        self.client_masks = [None for _ in self.clients]
        self.client_has_been_selected = [False for _ in self.clients]

    def pack(self, client_id, mtype=0, *args, **kwargs):
        if not self.client_has_been_selected[client_id]:
            return {'model': copy.deepcopy(self.model), 'init_weight': self.init_weight}
        else:
            return {'model': copy.deepcopy(self.model)}

    def iterate(self):
        self.selected_clients = self.sample()
        res = self.communicate(self.selected_clients)
        masks, models = res['mask'], res['model']
        self.model = self.aggregate(models, masks)
        for cid in self.selected_clients: self.client_has_been_selected[cid] = True
        return

    def aggregate(self, models: list, masks: list):
        w = [mi.state_dict() for mi in models]
        w_avg = copy.deepcopy(w[0])
        mask = [np.zeros_like(masks[0][j]) for j in range(len(masks[0]))]
        for i in range(len(masks)):
            for j in range(len(mask)):
                mask[j]+= masks[i][j]
        step = 0
        for key in w_avg.keys():
            if 'weight' in key:
                for i in range(1, len(w)):
                    w_avg[key] += w[i][key]
                w_avg[key] = torch.from_numpy(np.where(mask[step] < 1, 0, w_avg[key].cpu().numpy() / mask[step]))
                step += 1
            else:
                for i in range(1, len(w)):
                    w_avg[key] += w[i][key]
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
        self.mask = None

    def unpack(self, received_pkg):
        if 'init_weight' in received_pkg.keys(): self.init_weight = received_pkg['init_weight']
        train_model = received_pkg['model']
        if self.mask is None: self.mask = self.init_mask(train_model)
        self.mask_model(train_model, self.mask, train_model.state_dict())
        acc_before_train = self.test(train_model, 'val')['accuracy']
        if acc_before_train>self.prune_start_acc and self.current_prune_rate>self.prune_end_rate:
            # mask global model from initial weights when the accuracy is adeduately high and the prune rate is not adeduately low (i.e. the number of parameters is too large)
            # once the accuracy is low or the prune rate achieves the target rate, stop clipping
            self.prune_by_percentile(train_model, self.mask)
            self.current_prune_rate = self.current_prune_rate * (1 - self.prune_rate)
            self.mask_model(train_model, self.mask, self.init_weight)
        return train_model

    def pack(self, model, *args, **kwargs):
        self.mask_model(model, self.mask, model.state_dict())
        self.model = model
        return {'mask': self.mask, 'model': self.model}

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

    def mask_model(self, model, mask, state_dict):
        step = 0
        for name, param in model.named_parameters():
            if "weight" in name:
                weight_dev = param.device
                param.data = torch.from_numpy(mask[step] * state_dict[name].cpu().numpy()).to(weight_dev)
                step = step + 1
            if "bias" in name:
                param.data = state_dict[name]
        return

    def init_mask(self, model):
        return [np.ones_like(param.data.cpu().numpy()) for name, param in model.named_parameters() if 'weight' in name]

