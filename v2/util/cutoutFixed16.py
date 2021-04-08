import torch
import numpy as np


class CutoutF(object):
    """Randomly mask out one or more patches from an image.

    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            rand_center = np.random.randint(3)
            if rand_center == 0:
                y1 = 0
                y2 = self.length
                x1 = 0
                x2 = self.length
            elif rand_center == 1:
                y1 = 0
                y2 = self.length
                x1 = self.length
                x2 = w
            elif rand_center == 2:
                y1 = self.length
                y2 = h
                x1 = 0
                x2 = self.length
            else:
                y1 = self.length
                y2 = h
                x1 = self.length
                x2 = w

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img