import torchvision
import os
import torch.nn as nn
import torch.nn.functional as F
import flgo.benchmark
import torch

transform = torchvision.transforms.Compose(
    [torchvision.transforms.ToTensor(),
     torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.262))]
)

path = os.path.join(flgo.benchmark.data_root,  'CIFAR100')
train_data = torchvision.datasets.CIFAR100(root=path, train=True, download=True, transform=transform)
test_data = torchvision.datasets.CIFAR100(root=path, train=False, download=True, transform=transform)

criterion = torch.nn.CrossEntropyLoss()

def data_to_device(batch_data, device):
    return batch_data[0].to(device), batch_data[1].to(device)

def eval(model, data_loader, device):
    model.to(device)
    total_loss = 0.0
    num_correct = 0
    data_size = 0
    for batch_id, batch_data in enumerate(data_loader):
        batch_data = data_to_device(batch_data, device)
        outputs = model(batch_data[0])
        batch_mean_loss = criterion(outputs, batch_data[-1]).item()
        y_pred = outputs.data.max(1, keepdim=True)[1]
        correct = y_pred.eq(batch_data[-1].data.view_as(y_pred)).long().cpu().sum()
        num_correct += correct.item()
        total_loss += batch_mean_loss * len(batch_data[-1])
        data_size += len(batch_data[-1])
    return {'accuracy': 1.0 * num_correct / data_size, 'loss': total_loss / data_size}

def compute_loss(model, batch_data, device):
    batch_data = data_to_device(batch_data, device)
    outputs = model(batch_data[0])
    loss = criterion(outputs, batch_data[-1])
    return {'loss':loss}

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(1600, 384)
        self.fc2 = nn.Linear(384, 192)
        self.fc3 = nn.Linear(192, 100)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 1600)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def get_model():
    return Model()
