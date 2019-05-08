import sys
ros_path = '/opt/ros/kinetic/lib/python2.7/dist-packages'
if ros_path in sys.path:
    sys.path.remove(ros_path)

import numpy as np
import matplotlib.pyplot as plt
import argparse
import cv2


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


class Resize:
    def __init__(self, out_size):
        assert isinstance(out_size, (int, tuple))
        self.out_size = out_size

    def __call__(self, image):
        h, w = image.shape[0:2]
        if h < self.out_size[0] and w < self.out_size[1]:  # upsample
            image_out = cv2.resize(
                image, self.out_size, interpolation=cv2.INTER_LINEAR
            )  # faster with cv2.INTER_LINEAR prettier with cv2.INTER_CUBIC
        else:
            image_out = cv2.resize(
                image, self.out_size, interpolation=cv2.INTER_AREA)
        return image_out


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

    if plot:
        x_mesh, y_mesh = np.meshgrid(xedges, yedges)

        plt.figure(1)
        plt.gca().invert_yaxis()
        plt.pcolormesh(x_mesh, y_mesh, hist_log)
        plt.show()
    return {'hist': hist, 'hist_log': hist_log}


def convert_index_to_ab_value(index):
    helper = lambda x: -105 + 10 * x
    a = helper(index[0])
    b = helper(index[1])
    return (a, b)


def discretize_ab_bins(hist_log):
    non_zero_indices = np.argwhere(hist_log > -float('inf'))
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
    ab_bins = discretize_ab_bins(histogram_data['hist_log'])
    return ab_bins


def main():
    data_dir = parse_args()
    get_ab_bins_from_data(data_dir, plot=True)


if __name__ == '__main__':
    main()
