import torchvision
import os
import torch.nn as nn
import flgo.benchmark
import torch
transform = torchvision.transforms.Compose(
    [torchvision.transforms.ToTensor(),
     torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.262))]
)

path = os.path.join(flgo.benchmark.data_root,  'CIFAR10')
train_data = torchvision.datasets.CIFAR10(root=path, train=True, download=True, transform=transform)
test_data = torchvision.datasets.CIFAR10(root=path, train=False, download=True, transform=transform)

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
        self.embedder = nn.Sequential(
            nn.Conv2d(3, 64, 5),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 64, 5),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(1),
            nn.Linear(1600, 384),
            nn.ReLU(),
            nn.Linear(384, 192),
            nn.ReLU(),
        )
        self.fc = nn.Linear(192, 10)

    def forward(self, x):
        x = self.get_embedding(x)
        return self.fc(x)

    def get_embedding(self, x):
        return self.embedder(x)

def get_model():
    return Model()
