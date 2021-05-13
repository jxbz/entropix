import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from util.nero import Nero
from util.data import normalize_data

# from tqdm import tqdm
tqdm = lambda x: x


class SimpleNet(nn.Module):
    def __init__(self, depth, width):
        super(SimpleNet, self).__init__()

        self.initial = nn.Linear(784, width, bias=False)
        self.layers = nn.ModuleList([nn.Linear(width, width, bias=False) for _ in range(depth-2)])
        self.final = nn.Linear(width, 1, bias=False)

    def forward(self, x):
        x = self.initial(x)
        x = F.relu(x) * math.sqrt(2)
        for layer in self.layers:
            x = layer(x)
            x = F.relu(x) * math.sqrt(2)
        return self.final(x)


def train_network(train_loader, test_loader, depth, width, init_lr, decay):
    
    model = SimpleNet(depth, width).cuda()
    optim = Nero(model.parameters(), lr=init_lr)      
    lr_lambda = lambda x: decay**x
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda)

    train_acc_list = []
    train_acc = 0

    for epoch in range(100):
        model.train()

        for data, target in tqdm(train_loader):
            data, target = (data.cuda(), target.cuda())
            data, target = normalize_data(data, target)

            y_pred = model(data).squeeze()
            loss = (y_pred - target).norm()

            model.zero_grad()
            loss.backward()
            optim.step()

        lr_scheduler.step()

        model.eval()
        correct = 0
        total = 0

        for data, target in tqdm(train_loader):
            data, target = (data.cuda(), target.cuda())
            data, target = normalize_data(data, target)

            y_pred = model(data).squeeze()
            correct += (target.float() == y_pred.sign()).sum().item()
            total += target.shape[0]

        train_acc = correct/total
        train_acc_list.append(train_acc)

        if train_acc == 1.0: break

    model.eval()
    correct = 0
    total = 0

    for data, target in tqdm(test_loader):
        data, target = (data.cuda(), target.cuda())
        data, target = normalize_data(data, target)
        
        y_pred = model(data).squeeze()
        correct += (target.float() == y_pred.sign()).sum().item()
        total += target.shape[0]

    test_acc = correct/total
  
    return train_acc_list, test_acc, model
