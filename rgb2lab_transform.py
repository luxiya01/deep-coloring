import numpy as np
import cv2
from sklearn.neighbors import NearestNeighbors
from scipy.stats import norm


class RGB2LAB(object):
    def __init__(self, ab_bins, ab_bins_to_index):
        self.ab_bins = ab_bins
        self.ab_bins_to_index = ab_bins_to_index
        self.num_bins = len(ab_bins)
        self.nbrs = NearestNeighbors(
            n_neighbors=5, algorithm='ball_tree').fit(self.ab_bins)

    def __call__(self, sample):
        lab_image = cv2.cvtColor(sample, cv2.COLOR_BGR2LAB)
        # L channel is used as input
        l = lab_image[:, :, 0]

        # AB channels are used as ground truth
        ab = lab_image[:, :, 1:]
        w, h = ab.shape[0], ab.shape[1]

        ab_reshaped = ab.reshape(-1, 2)
        distances, indices = self.nbrs.kneighbors(ab_reshaped)

        # Output ab dim: 313 x w x h
        y_truth = np.zeros((self.num_bins, w, h))
        z_truth = np.zeros((self.num_bins, w, h))

        for i in range(w):
            for j in range(h):
                pixel_index = i * w + j
                assert ab[i, j] == ab_reshaped[pixel_index]

                true_ab = ab_reshaped[pixel_index]
                true_ab_index = self.ab_bins_to_index[true_ab]

                y_truth[true_ab_index, w, h] = 1
                z_truth[true_ab_index, w, h] = 1

                for nbr_idx, distance in zip(indices[pixel_index],
                                             distances[pixel_index]):
                    z_truth[nbr_idx, w, h] = norm.pdf(distance)
        return {'l': l, 'z_truth': z_truth, 'y_truth': y_truth}
