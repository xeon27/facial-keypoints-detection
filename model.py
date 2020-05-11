# Import libraries
import torch.nn as nn
import torch.nn.functional as F


class FKNet(nn.Module):
    def __init__(self):
        super(FKNet, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 32, 7)
        self.pool1 = nn.MaxPool2d(2)
        
        self.conv2 = nn.Conv2d(32, 64, 5, 2)
        self.norm = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2)
        
        self.conv3 = nn.Conv2d(64, 128, 3)
        self.pool3 = nn.MaxPool2d(4)
        
        self.fc1 = nn.Linear(128*6*6, 1000)
        self.dropout = nn.Dropout(0.2)
        
        self.fc2 = nn.Linear(1000, 136)
        
    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(self.norm(F.relu(self.conv2(x))))
        x = self.pool3(F.relu(self.conv3(x)))
        
        x = x.view((1, -1))
        x = self.dropout(self.fc1(x))
        x = self.fc2(x)
        
        return x
        