import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import sys
import network

ros_path = '/opt/ros/kinetic/lib/python2.7/dist-packages'
if ros_path in sys.path:
    sys.path.remove(ros_path)
import cv2
sys.path.append('/opt/ros/kinetic/lib/python2.7/dist-packages')


def read_all_images(path_to_data):
    """
    :param path_to_data: the file containing the binary images from the STL-10 dataset
    :return: an array containing all the images
    """
    with open(path_to_data, 'rb') as f:
        # read whole file in uint8 chunks
        everything = np.fromfile(f, dtype=np.uint8)

        # We force the data into 3x96x96 chunks, since the
        # images are stored in "column-major order", meaning
        # that "the first 96*96 values are the red channel,
        # the next 96*96 are green, and the last are blue."
        # The -1 is since the size of the pictures depends
        # on the input file, and this way numpy determines
        # the size on its own.

        images = np.reshape(everything, (-1, 3, 96, 96))

        # Now transpose the images into a standard image format
        # readable by, for example, matplotlib.imshow
        # You might want to comment this line or reverse the shuffle
        # if you will use a learning algorithm like CNN, since they like
        # their channels separated.
        images = np.transpose(images, (0, 3, 2, 1))
        return images


class Convert2lab:
    def __call__(self, images):
        ### Takes np.arrays and outputs np.arrays
        images_lab = np.zeros_like(images)
        for i in range(images.shape[0]):
            images_lab[i, :, :, :] = cv2.cvtColor(images[i, :, :, :],
                                                  cv2.COLOR_RGB2LAB)
        return images_lab


class Resize:
    def __init__(self, out_size):
        assert isinstance(out_size, tuple)
        self.out_size = out_size

    def __call__(self, images):
        ### Input torch tensor shape [n, channels, H, W], output is on same format
        print(images.shape)
        if len(images.shape) == 3 and images.shape[
                0] == 1:  # if only one image, channels should always have a dimension
            images_np = np.array([images.numpy()])  # 3D to 4D tensor
            print('snt')
            print(images_np.shape)
            images_np = np.transpose(
                images_np, (0, 2, 3, 1))  # opencv wants channels last
        else:
            images_np = images.numpy()
            images_np = np.transpose(
                images_np, (0, 2, 3, 1))  # opencv wants channels last

        print('resize')
        print(images_np.shape)
        h, w = images.shape[1:3]
        images_out = np.zeros((images_np.shape[0], self.out_size[0],
                               self.out_size[1], images_np.shape[3]))
        for i in np.arange(images_np.shape[0]):
            if h < self.out_size[0] and w < self.out_size[1]:  # upsample
                image_resized = cv2.resize(
                    images_np[i, :, :, :],
                    self.out_size,
                    interpolation=cv2.INTER_LINEAR
                )  # faster with cv2.INTER_LINEAR prettier with cv2.INTER_CUBIC
                if len(image_resized.shape) == 2:
                    image_resized = np.transpose(
                        np.array([image_resized]), (1, 2, 0))
                images_out[i, :, :, :]
            else:
                image_resized = cv2.resize(
                    images_np[i, :, :, :],
                    self.out_size,
                    interpolation=cv2.INTER_AREA)
                print(image_resized.shape)
                if len(image_resized.shape) == 2:
                    image_resized = np.transpose(
                        np.array([image_resized]), (1, 2, 0))
                images_out[i, :, :, :] = image_resized
        tensor_out = torch.from_numpy(np.transpose(images_out, (0, 1, 2, 3)))
        return tensor_out


def ab_histogram_from_dataset(dataset):
    #im_ab = dataset[0,:,:,1:]
    im_ab = dataset[:, :, :, 1:]
    im_ab = im_ab.astype(np.int16)
    im_ab_vec = np.reshape(
        im_ab,
        (np.prod(im_ab.shape[:3]), 2)) - 128  # 128 since colors are 8bit
    hist, xedges, yedges = np.histogram2d(
        im_ab_vec[:, 0],
        im_ab_vec[:, 1],
        bins=100,
        range=[[-110, 110], [-110, 110]])
    hist_log = np.log(hist / im_ab_vec.shape[0])
    # Plot
    x_mesh, y_mesh = np.meshgrid(xedges, yedges)
    plt.figure(1)
    plt.pcolormesh(x_mesh, y_mesh, hist_log)


def imshow_torch(image, figure=1):
    ### imshows torch tensors, image.shape should be [channel,H,W]
    plt.figure(figure)
    # opencv wants images like [H,W,channel]
    image_np_lab = np.transpose(image.numpy(), (1, 2, 0))
    
    print('lab shape')
    print(image_np_lab.shape)
    '''
    if image_np_lab.shape[2] == 1:  # Gray scale
        image_np_lab = image_np_lab[:, :, 0]
    '''
    image_np_rgb_all = cv2.cvtColor(image_np_lab, cv2.COLOR_LAB2RGB)
    plt.subplot(2,2,1)
    plt.imshow(image_np_rgb_all)
    plt.title('All channels')

    image_np_L = np.zeros_like(image_np_lab)
    image_np_L[:,:,0] = image_np_lab[:,:,0]
    image_np_rgb_L = cv2.cvtColor(image_np_L, cv2.COLOR_LAB2RGB)
    plt.subplot(2,2,2)
    plt.imshow(image_np_rgb_L)
    plot_image_channels(torchimage_np_rgb_L)
    plt.title('Only L channel')

    image_np_a = np.zeros_like(image_np_lab)
    image_np_a[:,:,1] = image_np_lab[:,:,1]
    image_np_rgb_a = cv2.cvtColor(image_np_a, cv2.COLOR_LAB2RGB)
    plt.subplot(2,2,3)
    plt.imshow(image_np_rgb_a)
    plt.title('Only a channel')

    image_np_b = np.zeros_like(image_np_lab)
    image_np_b[:,:,2] = image_np_lab[:,:,2]
    image_np_rgb_b = cv2.cvtColor(image_np_b, cv2.COLOR_LAB2RGB)
    plt.subplot(2,2,4)
    plt.imshow(image_np_rgb_b)
    plt.title('Only b channel')


def plot_image_channels(image, figure=1):
    image = image.numpy()
    print(image.shape)
    plt.figure(figure)
    plt.subplot(3,1,1)
    plt.plot(image[0,:,:])
    plt.title('L channel')

    plt.subplot(3,1,2)
    plt.plot(image[1,:,:])
    plt.title('a channel')

    plt.subplot(3,1,3)
    plt.plot(image[2,:,:])
    plt.title('b channel')
    

if __name__ == '__main__':
    data_dir = '/home/perrito/kth/DD2424/project/images/stl10_binary/train_X.bin'
    images = read_all_images(data_dir)
    color_conversion = Convert2lab()
    images_lab = color_conversion(images)
    images_lab = np.transpose(images_lab, (0, 3, 1, 2))  # torch format
    images_L = np.transpose(np.array([images_lab[:, 0, :, :]]), (1, 0, 2, 3))
    lab_torch = torch.from_numpy(images_lab)  # Images in torch format
    L_torch = torch.from_numpy(images_L).float()
    #ab_histogram_from_dataset(images_lab)

    test_set = L_torch[:1, :, :, :]
    test_set256 = F.interpolate(
        test_set, size=256, mode='bilinear', align_corners=False)

    net = network.Net()
    out_ab = net(test_set256.float())
    colorized_im = torch.cat((test_set256, out_ab), 1)
    
    colorized_im_np = colorized_im.detach().numpy()

    plot_image_channels(colorized_im.detach()[0,:,:,:], figure=1)
    imshow_torch(colorized_im.detach()[0, :, :, :], figure=2)

    imshow_torch(lab_torch[0, :, :, :], figure=3)
    plot_image_channels(lab_torch[0, :, :, :], figure=4)


    plt.show()



