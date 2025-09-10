import torch
import torch.nn as nn
import torch.nn.functional as F

class FNN(nn.Module):
    def __init__(self, loss_type, num_classes):
        super(FNN, self).__init__()
        self.loss_type = loss_type
        self.num_classes = num_classes
        self.fc1 = nn.Linear(28 * 28, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.tanh(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        if self.loss_type == "l2":
            return F.softmax(x, dim=1)
        else:
            return x

    def get_loss(self, output, target):
        if self.loss_type == 'ce':
            loss = nn.CrossEntropyLoss()(output, target)
        elif self.loss_type == 'l2':
            # Source: https://pytorch.org/docs/stable/generated/torch.nn.functional.one_hot.html
            target = F.one_hot(target, num_classes=self.num_classes).float()
            loss = nn.MSELoss()(output, target.float())
        else:
            print('Invalid loss type')
            exit(1)
        return loss
