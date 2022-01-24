from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

import copy

class Conv1(nn.Module):
    def __init__(self, outputs=2):
        super(Conv1, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(2*2*64, 128)
        self.fc2 = nn.Linear(128, outputs)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

class Conv2(nn.Module):
    '''
    - Input: 32x32x3.
    - Conv1 + pool: 15x15x16
    - Conv2 + pool: 6x6x32
    - Conv3 + pool: 2x2x64
    - FC: outputs
    
    '''
    
    def __init__(self, outputs=10):
        super(Conv2, self).__init__()
        self.conv1 = nn.Conv2d(3, 24, 5)
        self.conv2 = nn.Conv2d(24, 24, 3)
        self.conv3 = nn.Conv2d(24, 36, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(2*2*36, outputs)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.fc(torch.flatten(x, 1))
        x = F.log_softmax(x, dim=1)
        return x
        
        

class LogReg(nn.Module):
    def __init__(self, outputs=2):
        super(LogReg, self).__init__()
        self.fc1 = nn.Linear(8*8, outputs)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        output = F.log_softmax(x, dim=1)
        return output

class WrapperNet(nn.Module):
    def __init__(self, net):
        super(WrapperNet, self).__init__()
        self.net = net
    
    def forward(self, x):
        x = self.net(x)
        output = F.log_softmax(x, dim=1)
        return output
