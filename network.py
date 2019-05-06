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
        self.BN_momentum = 1e-3  # Batch normalization momentum
        self.color_bins = 313  # Number of color bins in ab colorspace
        # nn.Conv2d(a,b,c);  a input image channel, b output channels, cxc square convolution kernel
        ### Conv 1 ###
        self.bw_conv1_1 = nn.Conv2d(1, 64, 3, stride=1, padding=1, dilation=1)
        self.conv1_2 = nn.Conv2d(64, 64, 3, stride=2, padding=1, dilation=1)
        self.conv1_2norm = nn.BatchNorm2d(64, momentum=self.BN_momentum)
        ### Conv 2 ###
        self.conv2_1 = nn.Conv2d(64, 128, 3, stride=1, padding=1, dilation=1)
        self.conv2_2 = nn.Conv2d(128, 128, 3, stride=2, padding=1, dilation=1)
        self.conv2_2norm = nn.BatchNorm2d(128, momentum=self.BN_momentum)
        ### Conv 3 ###
        self.conv3_1 = nn.Conv2d(128, 256, 3, stride=1, padding=1, dilation=1)
        self.conv3_2 = nn.Conv2d(256, 256, 3, stride=1, padding=1, dilation=1)
        self.conv3_3 = nn.Conv2d(256, 256, 3, stride=2, padding=1, dilation=1)   
        self.conv3_3norm = nn.BatchNorm2d(256, momentum=self.BN_momentum)
        ### Conv 4 ###
        self.conv4_1 = nn.Conv2d(256, 512, 3, stride=1, padding=1, dilation=1)
        self.conv4_2 = nn.Conv2d(512, 512, 3, stride=1, padding=1, dilation=1)
        self.conv4_3 = nn.Conv2d(512, 512, 3, stride=1, padding=1, dilation=1)   
        self.conv4_3norm = nn.BatchNorm2d(512, momentum=self.BN_momentum)
        ### Conv 5 ###
        self.conv5_1 = nn.Conv2d(512, 512, 3, stride=1, padding=2, dilation=2)
        self.conv5_2 = nn.Conv2d(512, 512, 3, stride=1, padding=2, dilation=2)
        self.conv5_3 = nn.Conv2d(512, 512, 3, stride=1, padding=2, dilation=2)   
        self.conv5_3norm = nn.BatchNorm2d(512, momentum=self.BN_momentum)
        ### Conv 6 ###
        self.conv6_1 = nn.Conv2d(512, 512, 3, stride=1, padding=2, dilation=2)
        self.conv6_2 = nn.Conv2d(512, 512, 3, stride=1, padding=2, dilation=2)
        self.conv6_3 = nn.Conv2d(512, 512, 3, stride=1, padding=2, dilation=2)   
        self.conv6_3norm = nn.BatchNorm2d(512, momentum=self.BN_momentum)
        ### Conv 7 ###
        self.conv7_1 = nn.Conv2d(512, 512, 3, stride=1, padding=1, dilation=1)
        self.conv7_2 = nn.Conv2d(512, 512, 3, stride=1, padding=1, dilation=1)
        self.conv7_3 = nn.Conv2d(512, 512, 3, stride=1, padding=1, dilation=1)   
        self.conv7_3norm = nn.BatchNorm2d(512, momentum=self.BN_momentum)
        ### Conv 8 ###
        self.conv8_1 = nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1, dilation=1)  # The dilation should be on the input
        self.conv8_2 = nn.Conv2d(256, 256, 3, stride=1, padding=1, dilation=1)
        self.conv8_3 = nn.Conv2d(256, 256, 3, stride=1, padding=1, dilation=1)   
        self.conv8_color_bins = nn.Conv2d(256, self.color_bins, 1, stride=1, padding=0, dilation=1) 
        ### Softmax ###
        self.softmax8 = nn.Softmax2d() 
        ### Decoding ###
        self.conv8_ab = nn.Conv2d(self.color_bins, 2, 1, stride=1, padding=0, dilation=1)  


    def forward(self, in_data):
        ### Conv 1 ###
        print('in_data')
        print(in_data.shape)
        x = F.relu(self.bw_conv1_1(in_data))
        x = F.relu(self.conv1_2(x))
        x = self.conv1_2norm(x)
        print('Conv 1')
        print(x.shape)

        ### Conv 2 ###
        x = F.relu(self.conv2_1(x))
        x = F.relu(self.conv2_2(x))
        x = self.conv2_2norm(x)
        print('Conv 2')
        print(x.shape)
        ### Conv 3 ###
        x = F.relu(self.conv3_1(x))
        x = F.relu(self.conv3_2(x))
        x = F.relu(self.conv3_3(x))
        x = self.conv3_3norm(x)
        print('Conv 3')
        print(x.shape)
        ### Conv 4 ###
        x = F.relu(self.conv4_1(x))
        x = F.relu(self.conv4_2(x))
        x = F.relu(self.conv4_3(x))
        x = self.conv4_3norm(x)
        print('Conv 4')
        print(x.shape)
        ### Conv 5 ###
        x = F.relu(self.conv5_1(x))
        x = F.relu(self.conv5_2(x))
        x = F.relu(self.conv5_3(x))
        x = self.conv5_3norm(x)
        print('Conv 5')
        print(x.shape)
        ### Conv 6 ###
        x = F.relu(self.conv6_1(x))
        x = F.relu(self.conv6_2(x))
        x = F.relu(self.conv6_3(x))
        x = self.conv6_3norm(x)
        print('Conv 6')
        print(x.shape)
        ### Conv 7 ###
        x = F.relu(self.conv7_1(x))
        x = F.relu(self.conv7_2(x))
        x = F.relu(self.conv7_3(x))
        x = self.conv7_3norm(x)
        print('Conv 7')
        print(x.shape)
        ### Conv 8 ###
        x = F.relu(self.conv8_1(x))
        print('8.1')
        print(x.shape)
        x = F.relu(self.conv8_2(x))
        print(x.shape)
        x = F.relu(self.conv8_3(x))
        print(x.shape)
        x = F.relu(self.conv8_color_bins(x))
        print('Conv 8')
        print(x.shape)
        ### Softmax ###
        x = self.softmax8(x)
        ### Decoding ###
        x = self.conv8_ab(x)
        return x
'''
    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
'''
if __name__ == '__main__':

    net = Net()
    print(len(list(net.parameters())))
    in_data = torch.rand(1,1,256,256)
    out_data = net(in_data)
    print(out_data.shape)
