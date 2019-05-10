import torch
import torchvision  
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import sys  
import lab_distribution
from custom_transforms import RGB2LAB, ToTensor
from network import Net
from plot import *
from logger import Logger



ab_bins_dict = lab_distribution.get_ab_bins_from_data(
    '/home/perrito/kth/DD2424/project/images/stl10_binary/train_X.bin')
ab_bins, a_bins, b_bins = ab_bins_dict['ab_bins'], ab_bins_dict['a_bins'], ab_bins_dict['b_bins']

transform = transforms.Compose([RGB2LAB(ab_bins), ToTensor()])

trainset = torchvision.datasets.ImageFolder(
    root='tmp', transform=transform)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=2, shuffle=False, num_workers=2)

cnn = Net(ab_bins_dict)
criterion = cnn.loss
optimizer = optim.SGD(cnn.parameters(), lr=1e-3, momentum=0)
logger = Logger('./log')

for epoch in range(10):
    for i, data in enumerate(trainloader):
        inputs, labels = data
        lightness, z_truth, original = inputs['lightness'], inputs['z_truth'], inputs['original_lab_image']

        optimizer.zero_grad()
        outputs = cnn(lightness)
        if epoch == 0:
            colorized_im = torch.cat((lightness, outputs), 1)
            plot_image_channels(colorized_im.detach()[0, :, :, :], figure=20)
        loss = criterion(z_truth)
        print('loss')
        print(loss)
        loss.backward()
        optimizer.step()

        # Logging for tensorboardx
        info = { 'loss': loss }
        for tag, value in info.items():
            logger.scalar_summary(tag, value, i+1)


colorized_im = torch.cat((lightness, outputs), 1)

imshow_torch(colorized_im.detach()[0,:,:,:], figure=1)
plot_image_channels(colorized_im.detach()[0, :, :, :], figure=2)

imshow_torch(original[0,:,:,:], figure=3)
plot_image_channels(original[0, :, :, :], figure=4)

plt.show()