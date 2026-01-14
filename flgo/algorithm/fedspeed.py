import copy
from .fedbase import BasicServer
from .fedbase import BasicClient
import flgo.utils.fmodule as fmodule
import torch

class Server(BasicServer):
    def initialize(self, *args, **kwargs):
        """
        lmbd: 0.001, 0.01, 0.05, 0.1
        alpha: 0.0, 0.5, 0.75, 0.875, 0.9375, 1.0
        rho: 0.1
        """
        self.init_algo_para({"lmbd": 0.01, 'alpha':0.9, 'rho':0.1})
        self.aggregation_option = 'uniform'
class Client(BasicClient):
    def initialize(self, *args, **kwargs):
        self.local_grad_controller = self.server.model.zeros_like().to('cpu')
        self.register_cache_var('local_grad_controller')

    @fmodule.with_multi_gpus
    def train(self, model):
        model.train()
        global_model = copy.deepcopy(model)
        global_model.freeze_grad()
        self.local_grad_controller.to(self.device)
        his_sub_global = self.local_grad_controller - global_model
        optimizer = self.calculator.get_optimizer(model, lr=self.learning_rate, weight_decay=self.weight_decay+self.lmbd, momentum=self.momentum)
        for iter in range(self.num_steps):
            # get a batch of data
            optimizer.zero_grad()
            batch_data = self.get_batch_data()
            # Line 7 in Algo.1
            loss1 = self.calculator.compute_loss(model, batch_data)['loss']
            loss1.backward()
            grad1 = [p.grad for p in model.parameters()]
            # Line 8 in Algo.1
            grad1_norm = torch.norm(torch.cat([g.view(-1) for g in grad1 if g is not None]), 2)
            with torch.no_grad():
                for p, g in zip(model.parameters(), grad1):
                    p.data = p.data + self.rho * g / grad1_norm
            model.zero_grad()
            # Line 9 in Algo.1
            loss2 = self.calculator.compute_loss(model, batch_data)['loss']
            loss2.backward()
            grad2 = [p.grad for p in model.parameters()]
            model.zero_grad()
            with torch.no_grad():
                for p, g in zip(model.parameters(),grad1):
                    p.data = p.data - self.rho * g / grad1_norm
            # Line 10 in Algo.1
            quasi_grad = [(1.0-self.alpha)*g1+self.alpha*g2 if g1 is not None else None for g1, g2 in zip(grad1, grad2)]
            # Line 11 in Algo.1
            with torch.no_grad():
                for p, qg, hsgi in zip(model.parameters(), quasi_grad, his_sub_global.parameters()):
                    p.grad = qg - self.lmbd * hsgi
            if self.clip_grad > 0: torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=self.clip_grad)
            optimizer.step()
            # model_tmp.load_state_dict(copy.deepcopy(model.state_dict()))
        # Line 13 in Algo.1
        self.local_grad_controller = self.local_grad_controller + (model - global_model)
        # Line 14 in Algo.1
        model.load_state_dict((model + self.local_grad_controller).state_dict())
        self.local_grad_controller.to('cpu')
        return
