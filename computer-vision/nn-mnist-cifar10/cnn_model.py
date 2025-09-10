import timeit
from collections import OrderedDict

import torch
from torchvision import transforms, datasets

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import random_split

class Net(nn.Module):
    def __init__(self, in_channels):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        # Source: https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        if in_channels == 1:
            self.fc1 = nn.Linear(128 * 7 * 7, 512)  
        else:
            self.fc1 = nn.Linear(128 * 8 * 8, 512)  
        self.fc2 = nn.Linear(512, 256)  
        self.fc3 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.bn1(F.relu(self.conv1(x)))
        x = self.pool(x)
        x = self.bn2(F.relu(self.conv2(x)))
        x = self.pool(x)
        x = self.bn3(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x,dim=1)

def load_dataset(
        dataset_name: str,
):
    if dataset_name == "MNIST":
        full_dataset = datasets.MNIST('./data', train=True, download=True,
                                      transform=transforms.Compose([
                                          transforms.ToTensor(),
                                          transforms.Normalize((0.1307,), (0.3081,))]))

        train_dataset, valid_dataset = random_split(full_dataset, [48000, 12000])

    elif dataset_name == "CIFAR10":
        full_dataset = datasets.CIFAR10(root='./data', train=True, download=True,
                                        transform=transforms.Compose([
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))

        train_dataset, valid_dataset = random_split(full_dataset, [38000, 12000])

    else:
        raise Exception("Unsupported dataset.")

    return train_dataset, valid_dataset

def train(
        model,
        train_dataset,
        valid_dataset,
        device

):
    n_epochs = 10
    batch_size_train = 200
    batch_size_valid = 1000
    learning_rate = 0.01
    momentum = 0.5
    log_interval = 100

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size_train, shuffle=True)

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size= batch_size_valid, shuffle=False)

    train_losses = []
    train_counter = []
    validation_losses = []

    optimizer = optim.SGD(model.parameters(), lr=learning_rate,
                      momentum=momentum)

    def train(epoch):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))
                train_losses.append(loss.item())
                train_counter.append(
                    (batch_idx*64) + ((epoch-1)*len(train_loader.dataset)))
                
    def validation():
        model.eval()
        validation_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in valid_loader:
                data = data.to(device)
                target = target.to(device)
                output = model(data)
                validation_loss += F.nll_loss(output, target, size_average=False).item()
                pred = output.data.max(1, keepdim=True)[1]
                correct += pred.eq(target.data.view_as(pred)).sum()
        validation_loss /= len(valid_loader.dataset)
        validation_losses.append(validation_loss)
        print('\nValidation set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            validation_loss, correct, len(valid_loader.dataset),
            100. * correct / len(valid_loader.dataset)))
        
    validation()
    for epoch in range(1, n_epochs + 1):
        train(epoch)
        validation()
    torch.cuda.empty_cache()

    results = dict(
        model=model
    )

    return results

def CNN(dataset_name, device):

    #CIFAR-10 has 3 channels whereas MNIST has 1.
    if dataset_name == "CIFAR10":
        in_channels= 3
    elif dataset_name == "MNIST":
        in_channels = 1
    else:
        raise AssertionError(f'invalid dataset: {dataset_name}')

    model = Net(in_channels).to(device)

    train_dataset, valid_dataset = load_dataset(dataset_name)

    results = train(model, train_dataset, valid_dataset, device)

    return results

