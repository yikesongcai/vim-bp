import copy

from ..config import get_model
from flgo.utils.fmodule import FModule

class Model(FModule):
    def __init__(self):
        super().__init__()
        self.model = get_model()

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

def init_local_module(object):
    pass

def init_global_module(object):
    if 'Protocol' in object.__class__.__name__:
        object.model = Model().to(object.device)
    elif 'Client' in object.__class__.__name__:
        object.model = copy.deepcopy(object.protocal.model.to(object.device))