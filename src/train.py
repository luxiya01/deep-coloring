import torch
import torchvision
import torchvision.transforms as transforms
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
from tensorboardX import SummaryWriter
from torch.autograd import Variable

#data_dir = '/home/perrito/kth/DD2424/project/images/stl10_binary/train_X.bin'
data_dir = '../data/stl10/data/stl10_binary/train_X.bin'
ab_bins_dict = lab_distribution.get_ab_bins_from_data(data_dir)
ab_bins, a_bins, b_bins = ab_bins_dict['ab_bins'], ab_bins_dict[
    'a_bins'], ab_bins_dict['b_bins']

transform = transforms.Compose([RGB2LAB(ab_bins), ToTensor()])

trainset = torchvision.datasets.ImageFolder(
    root='tmp_red_bird', transform=transform)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=1, shuffle=False, num_workers=2)

cnn = Net(ab_bins_dict)
cnn.get_rarity_weights(data_dir)
criterion = cnn.loss
optimizer = optim.Adam(cnn.parameters(), weight_decay=.001)
# optimizer = optim.SGD(cnn.parameters(), lr=1e-2, momentum=0.9)
logger = Logger('./log')
logger.add_graph(cnn, image_size=96)

index = 0
for epoch in range(400):
    for i, data in enumerate(trainloader):
        inputs, labels = data
        lightness, z_truth, original = inputs['lightness'], inputs[
            'z_truth'], inputs['original_lab_image']

        optimizer.zero_grad()
        outputs = cnn(lightness)
        ab_outputs = cnn.decode_ab_values()

        colorized_im = torch.cat((lightness, ab_outputs), 1)
        #    plot_image_channels(colorized_im.detach()[0, :, :, :], figure=20)
        loss = criterion(z_truth)
        loss.backward()
        optimizer.step()

    # Logging loss to tensorboardx
    info = {'loss': loss}
    for tag, value in info.items():
        print(value.detach())
        logger.scalar_summary(tag, value.detach(), epoch )

    # Displaying Zhat
    logger.histogram_summary('Zhat', outputs.detach(), epoch)

    # Logging images to tensorboardx
    images = imshow_torch(colorized_im.detach()[0, :, :, :], figure=0)
    logger.add_image('output_image', torchvision.utils.make_grid(images),
                     epoch)

print('parameter exploration')
print(type(cnn.parameters()))
print(type(cnn.parameters().data))

colorized_im = torch.cat((lightness, ab_outputs), 1)
logger.histogram_summary('img', lightness, 0)

images = imshow_torch(colorized_im.detach()[0, :, :, :], figure=1)

#plot_image_channels(colorized_im.detach()[0, :, :, :], figure=2)

imshow_torch(original[0, :, :, :], figure=3)
#plot_image_channels(original[0, :, :, :], figure=4)

plt.show()
