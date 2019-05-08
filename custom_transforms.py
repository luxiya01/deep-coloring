import numpy as np
import cv2
from sklearn.neighbors import NearestNeighbors
from scipy.stats import norm
from matplotlib import pyplot as plt
import torch


class ToTensor(object):
    def __call__(self, sample):
        lightness, z_truth, original_lab_image = sample['lightness'], sample[
            'z_truth'], sample['original_lab_image']

        return {
            'lightness': torch.from_numpy(lightness),
            'z_truth': torch.from_numpy(z_truth),
            'original_lab_image': torch.from_numpy(original_lab_image)
        }


class RGB2LAB(object):
    def __init__(self, ab_bins):
        self.ab_bins = ab_bins
        self.ab_bins_to_index = {x: i for i, x in enumerate(self.ab_bins)}
        self.num_bins = len(ab_bins)
        self.nbrs = NearestNeighbors(
            n_neighbors=5, algorithm='ball_tree').fit(self.ab_bins)

    def __call__(self, sample):
        sample = np.asarray(sample, dtype=np.float32) / 255

        print('RGB2LAB is called!')

        # Image read in by PIL follows RGB convension, therefore the conversion
        # is RGB2LAB
        lab_image = cv2.cvtColor(sample, cv2.COLOR_RGB2LAB)
        # L channel is used as input
        l = lab_image[:, :, 0]

        # AB channels are used as ground truth
        ab = lab_image[:, :, 1:]

        w, h = ab.shape[0], ab.shape[1]

        ab_reshaped = ab.reshape(-1, 2)
        distances, indices = self.nbrs.kneighbors(ab_reshaped)

        # Output ab dim: num_bins x w x h
        #        z_truth = np.zeros((self.num_bins, w, h))

        num_pixels = w * h
        z_truth_reshaped = np.zeros((self.num_bins, num_pixels))
        z_truth_reshaped[indices,
                         np.arange(num_pixels).
                         reshape(num_pixels, 1)] = norm.pdf(distances)
        z_truth_reshaped = z_truth_reshaped.reshape((self.num_bins, w, h))

        #       for i in range(w):
        #           for j in range(h):
        #               pixel_index = i * w + j
        #               assert (ab[i, j] == ab_reshaped[pixel_index]).all()
        #
        #               for nbr_idx, distance in zip(indices[pixel_index],
        #                                            distances[pixel_index]):
        #                   z_truth[nbr_idx, i, j] = norm.pdf(distance)
        #               # self._plot(true_ab, z_truth)
        #       assert (z_truth == z_truth_reshaped).all()
        return {
            'lightness': l.reshape(1, w, h).astype(np.float32),
            'z_truth': z_truth_reshaped.astype(np.float32),
            'original_lab_image': lab_image.astype(np.float32)
        }

    def _plot(self, true_ab, z_truth):
        # Plot all ab_bins in our data domain as green dots(.)
        all_x, all_y = [val[0] for val in self.ab_bins], [
            val[1] for val in self.ab_bins
        ]
        plt.plot(all_x, all_y, 'g.')

        # Plot the true ab value of this pixel as a red circle(o)
        plt.plot(true_ab[0], true_ab[1], 'ro')

        # Plot the 5 nearest neighbors to the true ab value as blue stars(*)
        indices = [x[0] for x in np.argwhere(z_truth > 0)]
        x, y = [val[0] for val in np.array(self.ab_bins)[indices]], [
            val[1] for val in np.array(self.ab_bins)[indices]
        ]
        plt.plot(x, y, 'b*')
        plt.show()
