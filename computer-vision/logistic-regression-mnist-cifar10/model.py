import torch
import torchvision
from torch.autograd import Variable
from torch.utils.data import random_split
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torchvision.transforms as transforms
import numpy as np

def logistic_regression(dataset_name, device, batch_size_train=0, learning_rate=0, weight_decay=0.0005):
    n_epochs=10

    if (dataset_name == "MNIST"):
        batch_size_train = 256
        learning_rate = 0.005
        weight_decay = 0.0005
        MNIST_training = torchvision.datasets.MNIST('./MNIST_dataset/', train=True, download=True,
                                    transform=torchvision.transforms.Compose([
                                    torchvision.transforms.ToTensor(),
                                    torchvision.transforms.Normalize((0.1307,), (0.3081,))]))
        MNIST_training_set, MNIST_validation_set = random_split(MNIST_training, [48000, 12000])
        train_loader = torch.utils.data.DataLoader(MNIST_training_set,batch_size=batch_size_train, shuffle=True)
        validation_loader = torch.utils.data.DataLoader(MNIST_validation_set,batch_size=batch_size_train, shuffle=True)
        input_size = 28*28
    elif (dataset_name == "CIFAR10"):
        batch_size_train = 128
        learning_rate = 0.00001
        weight_decay = 0.0001
        # Source: https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
        transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        CIFAR10_training = torchvision.datasets.CIFAR10(root='./CIFAR10_dataset', train=True,
                                                download=True, transform=transform)
        CIFAR10_training_set, CIFAR10_validation_set = random_split(CIFAR10_training, [38000, 12000])
        train_loader = torch.utils.data.DataLoader(CIFAR10_training_set,
                                                batch_size=batch_size_train,
                                                shuffle=True, num_workers=2)
        validation_loader = torch.utils.data.DataLoader(CIFAR10_validation_set,
                                                        batch_size=batch_size_train,
                                                        shuffle=True, num_workers=2)
        input_size=3*32*32
    else: 
        print("Dataset name must be either MNIST or CIFAR10")
        exit(1)

    class LogisticRegression(nn.Module):
        def __init__(self):
            super(LogisticRegression, self).__init__()
            self.fc = nn.Linear(input_size, 10)

        def forward(self, x):
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            return x

    logistic_model = LogisticRegression().to(device)
    optimizer = optim.Adam(logistic_model.parameters(), lr=learning_rate)

    def train(epoch,data_loader,model,optimizer):
        for batch_idx, (data, target) in enumerate(data_loader):
            data = data.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            # Source: https://pytorch.org/docs/stable/generated/torch.nn.Softmax.html
            output = model(data).softmax(dim=1)
            # Source: https://pytorch.org/docs/stable/generated/torch.nn.functional.cross_entropy.html
            loss = F.cross_entropy(output, target) 
            l2_reg = 0
            for param in logistic_model.parameters():
                l2_reg += (param ** 2).sum()
            loss += weight_decay * 0.5 * l2_reg
            loss.backward()
            optimizer.step()
                
    def eval(data_loader,model,dataset):
        loss = 0
        correct = 0
        with torch.no_grad(): 
            for data, target in data_loader:
                data = data.to(device)
                target = target.to(device)
                output = model(data)
                pred = output.data.max(1, keepdim=True)[1]
                correct += pred.eq(target.data.view_as(pred)).sum()
                loss += F.cross_entropy(output, target, reduction='sum').item()
        loss /= len(data_loader.dataset)
        accuracy = 100. * correct / len(data_loader.dataset)
        return accuracy

    eval(validation_loader,logistic_model,"Validation")
    best_accuracy = 0
    for epoch in range(1, n_epochs + 1):
        train(epoch,train_loader,logistic_model,optimizer)
        validation_accuracy = eval(validation_loader,logistic_model,"Validation")
        if validation_accuracy > best_accuracy:
            best_accuracy = validation_accuracy

    results = dict(
        model=logistic_model,
        accuracy = best_accuracy
    )

    return results


def tune_hyper_parameter(dataset_name, target_metric, device):
    best_params = dict(
        learning_rate=None,
        batch_size_train=None,
        weight_decay=None
    )
    best_metric = 0

    for i in range(5):
        if (dataset_name == 'MNIST'):
            # Source: https://numpy.org/doc/stable/reference/random/generated/numpy.random.uniform.html
            learning_rate = 10 ** np.random.uniform(-3, -2)  
            batch_size_train = np.random.choice([128, 256, 512])  
            weight_decay = 10 ** np.random.uniform(-4, -3)
        else:
            learning_rate = 10 ** np.random.uniform(-5, -4)  
            batch_size_train = np.random.choice([64, 128, 256]) 
            weight_decay = 10 ** np.random.uniform(-4, -3) 

        result = logistic_regression(dataset_name, device, batch_size_train, learning_rate, weight_decay)
        validation_accuracy = result['accuracy']

        if validation_accuracy > best_metric:
            if (target_metric == "loss"):
                best_metric = (100 - validation_accuracy) / 100
            else:
                best_metric = validation_accuracy
            best_params['learning_rate'] = learning_rate
            best_params['batch_size_train'] = batch_size_train
            best_params['weight_decay'] = weight_decay

    return best_params, best_metric
