import torch
import torchvision  
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import sys  

'''
ros_path = '/opt/ros/kinetic/lib/python2.7/dist-packages'
if ros_path in sys.path:
    sys.path.remove(ros_path)
import cv2
sys.path.append('/opt/ros/kinetic/lib/python2.7/dist-packages')
'''
import cv2
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        #self.fc1 = nn.Linear(16 * 5 * 5, 120)
        #self.fc2 = nn.Linear(120, 84)
        #self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        #x = F.relu(self.fc1(x))
        #x = F.relu(self.fc2(x))
        #x = self.fc3(x)
        return x



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
        images_lab = np.zeros_like(images)
        for i in range(images.shape[0]):
            images_lab[i,:,:,:] = cv2.cvtColor(images[i,:,:,:], cv2.COLOR_BGR2LAB)
        return images_lab


class Resize:
    def __init__(self, out_size):
        assert isinstance(out_size, (int, tuple))
        self.out_size = out_size

    def __call__(self,image):
        h, w = image.shape[0:2]
        if h < self.out_size[0] and w < self.out_size[1]:  # upsample
            image_out = cv2.resize(image, self.out_size, interpolation=cv2.INTER_LINEAR)  # faster with cv2.INTER_LINEAR prettier with cv2.INTER_CUBIC  
        else:
            image_out = cv2.resize(image, self.out_size, interpolation=cv2.INTER_AREA)
        return image_out


def ab_histogram_dataset(dataset):
    #im_ab = dataset[0,:,:,1:]
    im_ab = dataset[:,:,:,1:]
    im_ab = im_ab.astype(np.int16)
    im_ab_vec = np.reshape(im_ab, (np.prod(im_ab.shape[:3]),2))-128  # 128 since colors are 8bit
    hist, xedges, yedges = np.histogram2d(im_ab_vec[:,0], im_ab_vec[:,1], bins=100, range=[[-110,110],[-110,110]])
    hist_log = np.log(hist/im_ab_vec.shape[0])
    x_mesh, y_mesh = np.meshgrid(xedges, np.flip(yedges))
    plt.figure(1)
    plt.pcolormesh(x_mesh, y_mesh, hist_log)
    


if __name__ == '__main__':
    net = Net()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    data_dir = '/home/perrito/kth/DD2424/project/images/stl10_binary/train_X.bin'
    images = read_all_images(data_dir)
    color_conversion = Convert2lab()
    images_lab = color_conversion(images)
    ab_histogram_dataset(images_lab)
    #image = torch.from_numpy(images[3,:,:,:])
    image = images[0,:,:,:]
    image_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    #plt.show()

    tf1 = Resize((512,512))
    tf2 = Resize((64,64))
    image2 = tf1(images_lab[0,:,:,:])
    image3 = tf2(image_lab)
    #image2 = cv2.resize(image, (80, 152))#, interpolation=cv2.INTER_CUBIC)
    #image3 = cv2.resize(image, (512, 512), interpolation=cv2.INTER_CUBIC)#, interpolation=cv2.INTER_CUBIC)
    #image2x = torchvision.transforms.Resize(image, (int(92*2), int(92*2))))
plt.figure(2)
plt.subplot(1,3,1)
plt.imshow(image)
plt.subplot(1,3,2)
plt.imshow(image2)
plt.subplot(1,3,3)
plt.imshow(image3)
plt.show()

trainloader = torch.utils.data.DataLoader(images, batch_size=4,
                                          shuffle=True, num_workers=2)
#image_tensor = torch.from_numpy(np.array([image], dtype='float64'))
#print(image_tensor.shape)
#image_tensor = image_tensor.permute(0,3,1,2)
#print(image_tensor.shape)
#output  = net(image_tensor)
