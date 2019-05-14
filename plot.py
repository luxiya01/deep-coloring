import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import network
import copy
import sys

ros_path = '/opt/ros/kinetic/lib/python2.7/dist-packages'
if ros_path in sys.path:
    sys.path.remove(ros_path)
import cv2
sys.path.append('/opt/ros/kinetic/lib/python2.7/dist-packages')


def imshow_torch(image, figure=1, plot=False):
    ### imshows torch tensors, image.shape should be [channel,H,W]

    image_np_lab = np.transpose(
        image.numpy(), (1, 2, 0))  # opencv wants images like [H,W,channel]

    # All channels
    image_np_rgb_all = cv2.cvtColor(image_np_lab, cv2.COLOR_LAB2RGB)
    
    # L channel
    image_np_L = np.zeros_like(image_np_lab)
    image_np_L[:, :, 0] = image_np_lab[:, :, 0]
    image_np_rgb_L = cv2.cvtColor(image_np_L, cv2.COLOR_LAB2RGB)

    # La channels
    image_np_a = copy.copy(image_np_lab)
    image_np_a[:, :, 2] = 0
    image_np_rgb_a = cv2.cvtColor(image_np_a, cv2.COLOR_LAB2RGB)

    # Lb channels
    image_np_b = copy.copy(image_np_lab)
    image_np_b[:, :, 1] = 0
    image_np_rgb_b = cv2.cvtColor(image_np_b, cv2.COLOR_LAB2RGB)

    # Plots
    if plot:
        plt.figure(figure)
        plt.subplot(2, 2, 1)
        plt.imshow(image_np_rgb_all)
        plt.title('All channels')

        plt.subplot(2, 2, 2)
        plt.imshow(image_np_rgb_L)
        plt.title('Only L channel')

        plt.subplot(2, 2, 3)
        plt.imshow(image_np_rgb_a)
        plt.title('Only L and a channel')

        plt.subplot(2, 2, 4)
        plt.imshow(image_np_rgb_b)
        plt.title('Only L and b channel')
    images = {
        'image_np_rgb_all':
        torch.from_numpy(np.transpose(image_np_rgb_all, (2, 0, 1))),
        'image_np_L':
        torch.from_numpy(np.transpose(image_np_rgb_L, (2, 0, 1))),
        'image_np_a':
        torch.from_numpy(np.transpose(image_np_rgb_a, (2, 0, 1))),
        'image_np_b':
        torch.from_numpy(np.transpose(image_np_rgb_b, (2, 0, 1)))
    }
    return torch.cat((images['image_np_rgb_all'], images['image_np_L'],
                      images['image_np_a'], images['image_np_b']), 2)


def plot_image_channels(image, figure=1):
    image = image.numpy()
    plt.figure(figure)
    plt.subplot(3, 1, 1)
    plt.plot(image[0, :, :])
    plt.title('L channel')

    plt.subplot(3, 1, 2)
    plt.plot(image[1, :, :])
    plt.title('a channel')

    plt.subplot(3, 1, 3)
    plt.plot(image[2, :, :])
    plt.title('b channel')
