import torch
import numpy as np
import copy
import torch.nn.functional as nnf

class Cutout_TRY(object):
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
        # only 4 random points
        xs = [w//4, w//2+w//4]
        ys = [h//4, h//2+h//4]

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.choice(ys)
            x = np.random.choice(xs)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img

class MANGO_TRY(object):
    def __init__(self, n_holes, length, model, device):
        self.model = model
        self.n_holes = n_holes
        self.length = length
        self.device = device

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.size(1)
        w = img.size(2)
        # only 4 random points
        xs = [w//4, w//2+w//4]
        ys = [h//4, h//2+h//4]
        
        res_img = img
        min_prob = -1
        
        for x in xs:
            for y in ys:
                temp_img = copy.deepcopy(img)
                mask = np.ones((h, w), np.float32)

                y1 = np.clip(y - self.length // 2, 0, h)
                y2 = np.clip(y + self.length // 2, 0, h)
                x1 = np.clip(x - self.length // 2, 0, w)
                x2 = np.clip(x + self.length // 2, 0, w)

                mask[y1: y2, x1: x2] = 0.

                mask = torch.from_numpy(mask)
                mask = mask.expand_as(temp_img)
                temp_img = temp_img * mask

                # predict
                temp_img = temp_img.view(1,3,32,32).to(self.device)
                pred = self.model(temp_img).to(self.device)
                if min_prob == -1: # initial prediction
                    value, index = nnf.softmax(pred, dim = 1).max(1)
                    min_prob = value
                    label = index[0]
                    continue
                softmax_prob = nnf.softmax(pred, dim=1)
                prob = softmax_prob[0][label]
                if prob > min_prob:
                    min_prob = prob
                    res_img = temp_img
        return res_img.view(3,32,32).to(self.device)