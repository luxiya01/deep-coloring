import sys
ros_path = '/opt/ros/kinetic/lib/python2.7/dist-packages'
if ros_path in sys.path:
    sys.path.remove(ros_path)

import numpy as np
import matplotlib.pyplot as plt
import argparse
import cv2
import scipy.ndimage.filters as fi
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm



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
        images = np.transpose(images, (0, 3, 2, 1)).astype(np.float32) / 255
        return images


class Convert2lab:
    def __call__(self, images):
        images_lab = np.zeros_like(images)
        for i in range(images.shape[0]):
            images_lab[i, :, :, :] = cv2.cvtColor(images[i, :, :, :],
                                                  cv2.COLOR_BGR2LAB)
        return images_lab


def ab_histogram_dataset(dataset, plot=False):
    im_ab = dataset[:, :, :, 1:]
    im_ab_vec = np.reshape(
        im_ab, (np.prod(im_ab.shape[:3]), 2))  # 128 since colors are 8bit
    hist, xedges, yedges = np.histogram2d(
        im_ab_vec[:, 0],
        im_ab_vec[:, 1],
        bins=(110 * 2) / 10,
        range=[[-110, 110], [-110, 110]])
    hist_log = np.log(hist / im_ab_vec.shape[0])
    p = hist/im_ab_vec.shape[0] 

    if plot:
        x_mesh, y_mesh = np.meshgrid(xedges, yedges)

        plt.figure(1)
        plt.gca().invert_yaxis()
        plt.pcolormesh(x_mesh, y_mesh, hist_log)
        #plt.show()
    return {'hist': hist, 'hist_log': hist_log, 'p': p}

def rarity_weights(p, sigma=5, lambda_=0.5):
    ### p is 2d matrix with probabilities
    p_tilde = fi.gaussian_filter(p, sigma)
    w_unm = 1/((1-lambda_) * p_tilde + lambda_/p.size)  # unnormalized and unmasked weights
    w_un = np.multiply(w_unm, p>1e-30)  # Delete the weights that aren't in gamut
    E_w = np.sum(np.multiply(w_un,p_tilde)) # expected value
    w = w_un - E_w + 1  # Normalized weights
    return w  # w are square matrix, bins that aren't in gamut are removed in get_rarity_weights()

def convert_index_to_ab_value(index):
    helper = lambda x: -105 + 10 * x
    a = helper(index[0])
    b = helper(index[1])
    return (a, b)


def discretize_ab_bins(matrix, threshold):
    non_zero_indices = np.argwhere(matrix > threshold)
    ab_bins = []
    a_bins = []
    b_bins = []
    for i in non_zero_indices:
        ab_val = convert_index_to_ab_value(i)
        a, b = ab_val
        a_bins.append(a)
        b_bins.append(b)
        ab_bins.append(ab_val)
    return {
        'ab_bins': np.array(ab_bins),
        'a_bins': np.array(a_bins),
        'b_bins': np.array(b_bins)
    }


def flatten_rarity_matrix(matrix, threshold):
    mask = matrix > threshold
    non_zero_indices = np.argwhere(mask>0)
    flat = np.zeros((1,np.count_nonzero(mask)))
    i = 0
    for ind in non_zero_indices:
        flat[0,i] = matrix[ind[0], ind[1]]
        i += 1
    return flat


def parse_args():
    parser = argparse.ArgumentParser(description='Parse data dir')
    parser.add_argument(
        '--data_dir', required=True, help='Path to the .bin data file')
    args = parser.parse_args()
    return args.data_dir


def get_ab_bins_from_data(data_dir, plot=False):
    images = read_all_images(data_dir)
    color_conversion = Convert2lab()
    images_lab = color_conversion(images)
    histogram_data = ab_histogram_dataset(images_lab, plot)
    ab_bins = discretize_ab_bins(histogram_data['hist_log'], -float('inf'))
    return ab_bins

def get_rarity_weights(data_dir, plot=False):
    data_dir = parse_args()
    images = read_all_images(data_dir)
    color_conversion = Convert2lab()
    images_lab = color_conversion(images)
    histogram_data = ab_histogram_dataset(images_lab, plot)
    w = rarity_weights(histogram_data['p'])
    w_bins = flatten_rarity_matrix(w, w.min()+1)
    return w_bins


def main():
    data_dir = parse_args()
    get_ab_bins_from_data(data_dir, plot=True)
    get_rarity_weights(data_dir, plot = False)


if __name__ == '__main__':
    main()






'''
def test():
    data_dir = parse_args()
    images = read_all_images(data_dir)
    color_conversion = Convert2lab()
    images_lab = color_conversion(images)
    histogram_data = ab_histogram_dataset(images_lab, plot=1)
    print(histogram_data['hist'].shape)
    w = rarity_weights(histogram_data['p'])
    print(w.shape)
    w_bins = discretize_ab_bins(w, w.min()+1)
    print('bins')
    print(w_bins['ab_bins'].shape)
    fig = plt.figure(30)
    ax = fig.gca(projection='3d')
    X = np.arange(w.shape[0])
    Y = np.arange(w.shape[1])
    X,Y = np.meshgrid(X,Y)
    surf = ax.plot_surface(X,Y,w, cmap=cm.coolwarm)
    plt.figure(10)
    plt.pcolormesh(w)
    plt.figure(12)
    plt.pcolormesh(w>w.min()+1)
    plt.figure(11)
    plt.pcolormesh(histogram_data['p']>1e-30)
    plt.show()
'''
