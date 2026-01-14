import torchvision
import os
import flgo.benchmark
import torch.nn as nn
import torch

transform = torchvision.transforms.Compose(
    [torchvision.transforms.ToTensor(),
     torchvision.transforms.Normalize((0.1307,), (0.3081,))]
)
path = os.path.join(flgo.benchmark.data_root, 'MNIST')
train_data = torchvision.datasets.MNIST(root=path, train=True, download=True, transform=transform)
test_data = torchvision.datasets.MNIST(root=path, train=False, download=True, transform=transform)

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
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(1),
            nn.Linear(3136, 512),
            nn.ReLU(),
        )
        self.head = nn.Linear(512, 10)

    def forward(self, x):
        x = self.encoder(x)
        x = self.head(x)
        return x

def get_model():
    return Model()