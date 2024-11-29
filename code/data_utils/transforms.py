import torch
import math
import numbers
import random
import numpy as np
import pdb
import time
from PIL import Image
from skimage import measure
import cv2
import scipy
from scipy.ndimage.interpolation import zoom


class ZoomOut(object):
    def __init__(self, size):
        self.size = np.array(size)

    def __call__(self, sample):
        sample['image'] = zoom(sample['image'], self.size / sample['image'].shape, order=1)
        return sample


class Normalize(object):
    def __init__(self):
        pass

    def __call__(self, sample):
        sample['image'] = np.expand_dims(np.array(sample['image']) / 255.0, 0).astype(np.float32)
        return sample


class LandMarkToMaskGaussianHeatMap(object):
    def __init__(self, R=20., img_size=(96,96,96), n_class=29, GPU=None):
        self.R = R
        self.GPU = GPU

        c_row = np.array([i for i in range(img_size[0])])
        c_matrix = np.stack([c_row] * img_size[1], 1)
        c_matrix = np.stack([c_matrix] * img_size[2], 2)
        c_matrix = np.stack([c_matrix] * n_class, 0)
        
        h_row = np.array([i for i in range(img_size[1])])
        h_matrix = np.stack([h_row] * img_size[0], 0)
        h_matrix = np.stack([h_matrix] * img_size[2], 2)
        h_matrix = np.stack([h_matrix] * n_class, 0)

        w_row = np.array([i for i in range(img_size[2])])
        w_matrix = np.stack([w_row] * img_size[0], 0)
        w_matrix = np.stack([w_matrix] * img_size[1], 1)
        w_matrix = np.stack([w_matrix] * n_class, 0)
        if GPU is not None:
            self.c_matrix = torch.tensor(c_matrix).float().to(self.GPU)
            self.h_matrix = torch.tensor(h_matrix).float().to(self.GPU)
            self.w_matrix = torch.tensor(w_matrix).float().to(self.GPU)

    def __call__(self, landmark):
        n_landmark = landmark.shape[1]
        batch_size = landmark.shape[0]

        if self.GPU is not None:
            mask = torch.sqrt(
                torch.pow(self.c_matrix - torch.tensor(np.expand_dims(np.expand_dims(landmark[:, :, 0:1], 3),4)).float().to(self.GPU), 2) +
                torch.pow(self.h_matrix - torch.tensor(np.expand_dims(np.expand_dims(landmark[:, :, 1:2], 3),4)).float().to(self.GPU), 2) +
                torch.pow(self.w_matrix - torch.tensor(np.expand_dims(np.expand_dims(landmark[:, :, 2:3], 3),4)).float().to(self.GPU), 2)
                ) <= self.R

            cur_heatmap = torch.exp(-((
                torch.pow(self.c_matrix - torch.tensor(np.expand_dims(np.expand_dims(landmark[:, :, 0:1], 3),4)).float().to(self.GPU), 2) +
                torch.pow(self.h_matrix - torch.tensor(np.expand_dims(np.expand_dims(landmark[:, :, 1:2], 3),4)).float().to(self.GPU), 2) +
                torch.pow(self.w_matrix - torch.tensor(np.expand_dims(np.expand_dims(landmark[:, :, 2:3], 3),4)).float().to(self.GPU), 2)
                ) / (0.2 * self.R * self.R)))
            heatmap = 2 * cur_heatmap * mask.float() + mask.float() - 1
        return heatmap
