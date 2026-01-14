"""
This module is designed for creating challenging federated scenarios like noisy features, poisoned data, and truncated dataset based on existing tasks.
Each decorator can be applied to an existing runner by directly invoking the decorator and passing the runner as an argument. For example,
>>> runner = flgo.init(task, fedavg, option) # create runner
>>> import flgo.decorator as fd
>>> decorator = fd.TrainingDataReducer(0.8)
>>> decorator(runner)
>>> runner.run()
"""
import os.path
import shutil
import numpy as np
from toolz import isiterable
import torch.utils.data as tud

class BasicDecorator:
    """
    Basic decorator. Each customized decorator should inherent from this class and overwrite the __call__ method.
    Remark: it's recommended to call self.register_runner at the end of each __call__ method. This will create an
    additional directory to save the records of the decorated task, preventing the original records with the same
    name to be overwritten.
    """
    def register_runner(self, runner):
        # redirect task path for outputs
        original_task = runner.gv.logger.option['task']
        new_task = os.path.join(original_task, str(self))
        if not os.path.exists(new_task):
            os.makedirs(os.path.join(new_task, 'record'))
            shutil.copy(os.path.join(original_task, 'info'), new_task)
        runner.option['task'] = runner.option['task'] + str(self)
        runner.gv.logger.option['task'] = new_task
        runner.gv.logger.task_path = new_task

class ClientRemover(BasicDecorator):
    """
    Exclude partial clients from the training process

    Args:
        remove_idxs (list): the indices of clients to be excluded
    """
    def __init__(self, remove_idx=[]):
        if not isiterable(remove_idx): remove_idx=[remove_idx]
        self.remove_idx = remove_idx
        self.removed_clients = []
        self.preserved_clients = None

    def __call__(self, runner, *args, **kwargs):
        clients = runner.clients
        if self.remove_idx!=[]:
            assert max(self.remove_idx) < len(clients)
            self.removed_clients = [runner.clients[eid] for eid in self.remove_idx]
            self.preserved_clients = [runner.clients[cid] for cid in range(len(runner.clients)) if cid not in self.remove_idx]
            runner.reset_clients(self.preserved_clients)
        else:
            self.removed_clients = []
            self.preserved_clients = runner.clients
        self.register_runner(runner)
        return

    def __str__(self):
        return f"ClientRemover{len(self.removed_clients)}"

class RandomClientRemover(BasicDecorator):
    """
    Randomly exclude some clients from training process

    Args:
        remove_ratio (float): the ratio of clients to be excluded
    """
    def __init__(self, remove_ratio:float=0.1):
        self.remove_ratio = max(min(remove_ratio, 1.0), 0.0)
        self.removed_clients = []
        self.preserved_clients = None

    def __call__(self, runner, *args, **kwargs):
        num_clients = len(runner.clients)
        remove_idxs = np.random.choice(list(range(num_clients)), size=self.remove_ratio, replace=False).tolist()
        self.removed_clients = [runner.clients[eid] for eid in remove_idxs]
        self.preserved_clients = [runner.clients[cid] for cid in range(len(runner.clients)) if cid not in remove_idxs]
        self.register_runner(runner)

    def __str__(self):
        return f"RandomClientRemover{self.remove_ratio}"

class TrainingDataReducer(BasicDecorator):
    """
    Reduce training data at each client side

    Args:
        preserve_ratio (float): the ratio of training data to be preserved
    """
    def __init__(self, preserve_ratio=0.5):
        self.preserve_ratio = preserve_ratio

    def __call__(self, runner, *args, **kwargs):
        for c in runner.clients:
            new_train_data = tud.Subset(c.train_data, np.random.choice(list(range(len(c.train_data))), size=max(1, int(len(c.train_data) * self.preserve_ratio)), replace=False).tolist())
            c.set_data(new_train_data, 'train')
        self.register_runner(runner)

    def __str__(self):
        return f"TrainingDataReducer{self.preserve_ratio}"

class TrainingDataTruncation(BasicDecorator):
    """
    Truncating training data at each client side

    Args:
        max_size (int): the maximum size of training data
    """
    def __init__(self, max_size: int=1000):
        self.max_size = max_size

    def __call__(self, runner, *args, **kwargs):
        for c in runner.clients:
            if len(c.train_data)>self.max_size:
                new_train_data = tud.Subset(c.train_data, np.random.choice(list(range(len(c.train_data))), size=self.max_size, replace=False).tolist())
                c.set_data(new_train_data, 'train')
        self.register_runner(runner)

    def __str__(self):
        return f"TrainingDataTruncation{self.max_size}"

class SequentialDecorator(BasicDecorator):
    """
    Combine several decorators into a single decorator

    Args:
        decorators (list): the list of decorators
    """
    def __init__(self, decorators):
        self.decorators = decorators

    def __call__(self, runner):
        for dec in self.decorators:
            dec(runner)
        self.register_runner(runner)

    def __str__(self):
        return f"Seq_{'-'.join([str(d) for d in self.decorators])}"

