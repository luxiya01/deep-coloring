import torch
import torchvision  
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import sys  







class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # nn.Conv2d(a,b,c);  a input image channel, b output channels, cxc square convolution kernel
        ### Conv 1 ###
        self.bw_conv1_1 = nn.Conv2d(1, 64, 3, stride=1, padding=1, dilation=1)
        self.conv1_2 = nn.Conv2d(1, 64, 3, stride=2, padding=1, dilation=1)
        self.conv1_2norm = nn.BatchNorm2d(64, momentum=1e-3)

    def forward(self, in_data):
        ### Conv 1 ###
        x = F.relu(self.bw_conv1_1(in_data))
        x = F.relu(self.bw_conv1_1(x))
        x = self.conv1_2norm(x)

        # If the size is a square you can only specify a single number
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


net = Net()
print(net)