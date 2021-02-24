import torch
import torch.nn.functional as nnf
import numpy as np


class Fixed_MNG(object):
    """mask out one or more patches from an image.

    Args:
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, length, model, n_masks = 4):
        self.model = model
        self.n_masks = n_masks
        #self.treshold = 1 ##########################TO BE SET??????
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        # print("start")
        h = img.size(1)
        w = img.size(2)

        masks = np.ones((self.n_masks, h, w), np.float32)

        y1 = np.clip(0, 0, h)
        y2 = np.clip(self.length, 0, h)
        y3 = np.clip(h, 0, h)
        x1 = np.clip(0, 0, w)
        x2 = np.clip(self.length, 0, w)
        x3 = np.clip(w, 0, w)
        
        masks[0][y1: y2, x1: x2] = 0.   ########or any other colors for mask
        masks[1][y1: y2, x2: x3] = 0.
        masks[2][y2: y3, x1: x2] = 0.
        masks[3][y2: y3, x2: x3] = 0.

        masks = torch.from_numpy(masks)

        with torch.no_grad(): 
            pred = self.model(img.view(1,3,32,32))
    
        value, index = nnf.softmax(pred, dim = 1).max(1)
        
        min_prob = value
        label = index[0]
        res = img
        # print("before loop")
        # print("before pred1")
        for mask in masks:
            mask = mask.expand_as(img)
            masked_img = img * mask.cuda()

            
            #####building child probability
            with torch.no_grad():
                pred = self.model(masked_img.view(1,3,32,32))
            # print("2")
            softmax_prob = nnf.softmax(pred, dim = 1)
            # print("3")
            prob = softmax_prob[0][label]

            if prob < min_prob: ######treshold comes here
                min_prob = prob ###??pointer
                res = masked_img
        # print("end")
        return res