import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import sys
import lab_distribution as lab_dist


class Net(nn.Module):
    # Input should be [n, 1, 256, 256] torch tensor
    def __init__(self, ab_bins_dict):
        super(Net, self).__init__()
        self.a_bins = ab_bins_dict['a_bins']
        self.b_bins = ab_bins_dict['b_bins']
        self.BN_momentum = 1e-3  # Batch normalization momentum
        self.color_bins = 225  # Number of color bins in ab colorspace
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
        self.conv8_1 = nn.ConvTranspose2d(
            512, 256, 4, stride=2, padding=1,
            dilation=1)  # The dilation should be on the input
        self.conv8_2 = nn.Conv2d(256, 256, 3, stride=1, padding=1, dilation=1)
        self.conv8_3 = nn.Conv2d(256, 256, 3, stride=1, padding=1, dilation=1)
        self.conv8_color_bins = nn.Conv2d(
            256, self.color_bins, 1, stride=1, padding=0, dilation=1)
        ### Softmax ###
        self.softmax8 = nn.Softmax2d()
        ### Decoding and upsampling ###
        self.conv8_ab = nn.Conv2d(
            self.color_bins, 2, 1, stride=1, padding=0, dilation=1)

    def forward(self, in_data):
        ### Conv 1 ###
        x = F.relu(self.bw_conv1_1(in_data))
        x = F.relu(self.conv1_2(x))
        x = self.conv1_2norm(x)
        ### Conv 2 ###
        x = F.relu(self.conv2_1(x))
        x = F.relu(self.conv2_2(x))
        x = self.conv2_2norm(x)
        ### Conv 3 ###
        x = F.relu(self.conv3_1(x))
        x = F.relu(self.conv3_2(x))
        x = F.relu(self.conv3_3(x))
        x = self.conv3_3norm(x)
        ### Conv 4 ###
        x = F.relu(self.conv4_1(x))
        x = F.relu(self.conv4_2(x))
        x = F.relu(self.conv4_3(x))
        x = self.conv4_3norm(x)
        ### Conv 5 ###
        x = F.relu(self.conv5_1(x))
        x = F.relu(self.conv5_2(x))
        x = F.relu(self.conv5_3(x))
        x = self.conv5_3norm(x)
        ### Conv 6 ###
        x = F.relu(self.conv6_1(x))
        x = F.relu(self.conv6_2(x))
        x = F.relu(self.conv6_3(x))
        x = self.conv6_3norm(x)
        ### Conv 7 ###
        x = F.relu(self.conv7_1(x))
        x = F.relu(self.conv7_2(x))
        x = F.relu(self.conv7_3(x))
        x = self.conv7_3norm(x)
        ### Conv 8 ###
        x = F.relu(self.conv8_1(x))
        x = F.relu(self.conv8_2(x))
        x = F.relu(self.conv8_3(x))
        x = F.relu(self.conv8_color_bins(x))

        ### Upsample###
        x = F.interpolate(x, size=96, mode='bilinear', align_corners=False)

        ### Softmax ###
        self.Zhat = self.softmax8(x)

        ### Decoding ###
        #        ab = self.decode_ab_values()
        #        lab = self.decode_final_colorful_image(in_data, ab)
        return self.Zhat

    def decode_ab_values(self):
        ### Decoding ###
        ind = torch.argmax(self.Zhat, dim=1)
        a, b = torch.from_numpy(self.a_bins[ind]), torch.from_numpy(
            self.b_bins[ind])
        a = torch.reshape(
            a, (a.shape[0], 1, a.shape[1], a.shape[2])
        )  # a has originally shape [samples,H,W], has to be [samples,channels,H,W]
        b = torch.reshape(b, (b.shape[0], 1, b.shape[1], b.shape[2]))
        ab = torch.cat((a, b), 1).float()

        print('Decoding done \n \n')
        return ab

    def decode_final_colorful_image(self, l, ab):
        lab = torch.cat((l, ab), 1)
        print('ab shape: ', ab.shape)
        print('l shape: ', l.shape)
        print('lab shape: ', lab.shape)
        return lab

    def loss(self, Z):
        q_star = torch.argmax(Z, 1)
        v = self.rarity_weights[0, q_star].float()
        #        l = -1 / Z.shape[0] * torch.sum(v * torch.sum(Z * torch.log(self.Zhat + 1e-30), (1)))

        #       print('q_star: ', q_star, q_star.shape)
        # print('v: ', v, v.shape)
        #       print('Z: ', Z, Z.shape)
        # print('Zhat: ', self.Zhat, self.Zhat.shape)
        #
        log_zhat = torch.log(self.Zhat + 1e-40)
        #       print('log_zhat: ', log_zhat, log_zhat.shape)
        #       print('max log_zhat: ', log_zhat.max())
        #       print('max log_zhat: ', log_zhat.min())
        #       #
        z_times_log_zhat = Z * log_zhat
        #       print('z_times_log_zhat: ', z_times_log_zhat, z_times_log_zhat.shape)
        #       print('max z_times_log_zhat: ', z_times_log_zhat.max())
        #       print('min z_times_log_zhat: ', z_times_log_zhat.min())
        #       #
        sum_z_times_log_zhat = torch.sum(z_times_log_zhat, 1)
        #       print('sum_z_times_log_zhat: ', sum_z_times_log_zhat,
        #             sum_z_times_log_zhat.shape)
        #       print('max sum_z_times_log_zhat: ', sum_z_times_log_zhat.max())
        #       print('min sum_z_times_log_zhat: ', sum_z_times_log_zhat.min())
        #       #
        v_times_sum = v * sum_z_times_log_zhat
        #       print('v_times_sum: ', v_times_sum, v_times_sum.shape)
        #       print('max v_times_sum: ', v_times_sum.max())
        #       print('min v_times_sum: ', v_times_sum.min())
        #       #
        sum_v_times_sum = torch.sum(v_times_sum)
        #       #
        #       print('-----------')
        #       print('sum_v_times_sum: ', sum_v_times_sum, sum_v_times_sum.shape)
        #       print('-----------')
        #       #
        l = -1 / (Z.shape[0] * Z.shape[2] * Z.shape[3]) * sum_v_times_sum
        #       print('l', l, l.shape)
        return l

    def get_rarity_weights(self, data_dir):
        self.rarity_weights = torch.from_numpy(
            lab_dist.get_rarity_weights(
                data_dir))  # This guy could be called every batch if we want


if __name__ == '__main__':

    net = Net()
    print(len(list(net.parameters())))
    in_data = torch.rand(1, 1, 256, 256)
    out_data = net(in_data)
    target = torch.rand(1, 2, 256, 256)

    print(out_data.shape)
    print(target.shape)

    criterion = nn.CrossEntropyLoss()
    criterion = nn.MSELoss()
    loss = criterion(out_data, target)
    print(loss.grad_fn)

    print(out_data.shape)
