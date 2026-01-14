"""
train_data (torch.utils.data.Dataset),
test_data (torch.utils.data.Dataset),
and the model (torch.nn.Module) should be implemented here.

"""
import torch.nn as nn
import torch.utils.data
train_data = None
val_data = None
test_data = None

def data_to_device(batch_data, device):
    raise NotImplementedError

def eval(model: nn.Module, data_loader: torch.utils.data.Dataset, device) -> dict:
    raise NotImplementedError

def compute_loss(batch_data, model:nn.Module, device) -> dict:
    raise NotImplementedError

def get_model(*args, **kwargs) -> torch.nn.Module:
    raise NotImplementedError