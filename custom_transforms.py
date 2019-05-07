import numpy as np
import cv2
from sklearn.neighbors import NearestNeighbors
from scipy.stats import norm
from matplotlib import pyplot as plt
import torch


class ToTensor(object):
    def __call__(self, sample):
        lightness, z_truth = sample['lightness'], sample['z_truth']

        return {
            'lightness': torch.from_numpy(lightness),
            'z_truth': torch.from_numpy(z_truth)
        }


class RGB2LAB(object):
    def __init__(self, ab_bins):
        self.ab_bins = ab_bins
        self.ab_bins_to_index = {x: i for i, x in enumerate(self.ab_bins)}
        self.num_bins = len(ab_bins)
        self.nbrs = NearestNeighbors(
            n_neighbors=5, algorithm='ball_tree').fit(self.ab_bins)

    def __call__(self, sample):
        sample = np.asarray(sample)
        print('RGB2LAB is called!')

        # Image read in by PIL follows RGB convension, therefore the conversion
        # is RGB2LAB
        lab_image = cv2.cvtColor(sample, cv2.COLOR_RGB2LAB)
        # L channel is used as input
        l = lab_image[:, :, 0].astype(np.float32)

        # AB channels are used as ground truth
        ab = lab_image[:, :, 1:].astype(np.int16) - 128
        w, h = ab.shape[0], ab.shape[1]

        ab_reshaped = ab.reshape(-1, 2)
        distances, indices = self.nbrs.kneighbors(ab_reshaped)

        # Output ab dim: num_bins x w x h
        y_truth = np.zeros((self.num_bins, w, h))
        z_truth = np.zeros((self.num_bins, w, h))

        for i in range(w):
            for j in range(h):
                pixel_index = i * w + j
                assert (ab[i, j] == ab_reshaped[pixel_index]).all()

                true_ab = ab_reshaped[pixel_index]

                for nbr_idx, distance in zip(indices[pixel_index],
                                             distances[pixel_index]):
                    z_truth[nbr_idx, i, j] = norm.pdf(distance)
                # self._plot(true_ab, z_truth)
        return {'lightness': l.reshape(1, w, h), 'z_truth': z_truth}

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
