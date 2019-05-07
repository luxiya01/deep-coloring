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
            images_lab[i,:,:,:] = cv2.cvtColor(images[i,:,:,:], cv2.COLOR_BGR2LAB)
        return images_lab


class Resize:
    def __init__(self, out_size):
        assert isinstance(out_size, tuple)
        self.out_size = out_size

    def __call__(self,images):
        ### Input torch tensor shape [n, channels, H, W], output is on same format
        print(images.shape)
        if len(images.shape) == 3 and images.shape[0] == 1:  # if only one image, channels should always have a dimension
            images_np = np.array([images.numpy()])  # 3D to 4D tensor
            print('snt')
            print(images_np.shape)
            images_np = np.transpose(images_np, (0,2,3,1))  # opencv wants channels last
        else:
            images_np = images.numpy()
            images_np = np.transpose(images_np, (0,2,3,1))  # opencv wants channels last

        print('resize')
        print(images_np.shape)
        h, w = images.shape[1:3]
        images_out = np.zeros((images_np.shape[0], self.out_size[0], self.out_size[1], images_np.shape[3]))
        for i in np.arange(images_np.shape[0]):
            if h < self.out_size[0] and w < self.out_size[1]:  # upsample
                image_resized = cv2.resize(images_np[i,:,:,:], self.out_size, interpolation=cv2.INTER_LINEAR)  # faster with cv2.INTER_LINEAR prettier with cv2.INTER_CUBIC  
                if len(image_resized.shape) == 2:
                    image_resized = np.transpose(np.array([image_resized]),(1,2,0))
                images_out[i,:,:,:] 
            else:
                image_resized = cv2.resize(images_np[i,:,:,:], self.out_size, interpolation=cv2.INTER_AREA)  
                print(image_resized.shape)
                if len(image_resized.shape) == 2:
                    image_resized = np.transpose(np.array([image_resized]),(1,2,0))
                images_out[i,:,:,:] = image_resized
        tensor_out = torch.from_numpy(np.transpose(images_out, (0,1,2,3)))
        return tensor_out


def ab_histogram_from_dataset(dataset):
    #im_ab = dataset[0,:,:,1:]
    im_ab = dataset[:,:,:,1:]
    im_ab = im_ab.astype(np.int16)
    im_ab_vec = np.reshape(im_ab, (np.prod(im_ab.shape[:3]),2))-128  # 128 since colors are 8bit
    hist, xedges, yedges = np.histogram2d(im_ab_vec[:,0], im_ab_vec[:,1], bins=100, range=[[-110,110],[-110,110]])
    hist_log = np.log(hist/im_ab_vec.shape[0])
    # Plot
    x_mesh, y_mesh = np.meshgrid(xedges, yedges)
    plt.figure(1)
    plt.pcolormesh(x_mesh, y_mesh, hist_log)

def imshow_torch(image, figure=1):
    ### imshows torch tensors, image.shape should be [channel,H,W]
    plt.figure(figure)
    image_np = np.transpose(image.numpy(), (1,2,0))
    if image_np.shape[2] == 1:  # Gray scale
        image_np = image_np[:,:,0]
    plt.imshow(image_np/255)



if __name__ == '__main__':
    data_dir = '/home/perrito/kth/DD2424/project/images/stl10_binary/train_X.bin'
    images = read_all_images(data_dir)
    color_conversion = Convert2lab()
    images_lab = color_conversion(images)
    images_lab = np.transpose(images_lab,(0,3,1,2))  # torch format
    images_L = np.transpose(np.array([images_lab[:,0,:,:]]),(1,0,2,3))
    lab_torch = torch.from_numpy(images_lab)  # Images in torch format
    L_torch = torch.from_numpy(images_L).float()
    #ab_histogram_from_dataset(images_lab)

    test_set = L_torch[:3,:,:,:]
    test_set256 = F.interpolate(test_set, size=256, mode='bilinear', align_corners=False)

    net = network.Net()
    out_ab = net(test_set256.float())
    print(out_ab.shape)
    print(test_set256.shape)
    colorized_im = torch.cat((test_set256, out_ab), 1)
    print(colorized_im.shape)
    imshow_torch(colorized_im.detach()[0,:,:,:])

    '''
    image3_L = torch.from_numpy(np.array([[image3[:,:,0]]]).astype(np.double))
    net = network.Net()
    image3_ab = net(image3_L.float())
    print('types')
    birb_L_np = image3_L.numpy()
    birb_ab_np = image3_ab.detach().numpy()
    print(birb_L_np.shape)
    birb_L_np =  cv2.resize(birb_L_np[0,:,:,:], (64,64), interpolation=cv2.INTER_LINEAR)
    print(type(image3_L.numpy()))
    print(birb_L_np.shape)
    print(type(image3_ab.detach().numpy()))
    print(birb_ab_np.shape)
    
    birb = np.concatenate(image3_L.numpy(), image3_ab.detach().numpy())
    '''
    plt.show()

#image_tensor = torch.from_numpy(np.array([image], dtype='float64'))
#print(image_tensor.shape)
#image_tensor = image_tensor.permute(0,3,1,2)
#print(image_tensor.shape)
#output  = net(image_tensor)
