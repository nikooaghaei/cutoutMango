import torch
import torch.nn.functional as nnf
import numpy as np


class Non_rec_MNG(object):
    """mask out one or more patches from an image.

    Args:
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, length, model, n_masks = 4):
        self.model = model
        self.n_masks = n_masks
        #self.treshold = 1 ##########################TO BE SET??????
        self.init_length = length

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
        mask_len = self.init_length

        y1 = np.clip(0, 0, h)
        y2 = np.clip(mask_len, 0, h)
        y3 = np.clip(h, 0, h)
        x1 = np.clip(0, 0, w)
        x2 = np.clip(mask_len, 0, w)
        x3 = np.clip(w, 0, w)

        with torch.no_grad(): 
            pred = self.model(img.view(1,3,32,32))
        
        value, index = nnf.softmax(pred, dim = 1).max(1)
        
        min_prob = value
        label = index[0]
        res = img
        while(mask_len > 0):    ####change to decide how deep to go


            masks = np.ones((self.n_masks, h, w), np.float32)
            
            masks[0][y1: y2, x1: x2] = 0.   ########or any other colors for mask
            masks[1][y1: y2, x2: x3] = 0.
            masks[2][y2: y3, x1: x2] = 0.
            masks[3][y2: y3, x2: x3] = 0.

            masks = torch.from_numpy(masks)

            expanding_node = -1     ###referring to original img
            
            for m, mask in enumerate(masks):
                mask = mask.expand_as(img)
                masked_img = img * mask.cuda()

                #####building child probability
                with torch.no_grad():
                    pred = self.model(masked_img.view(1,3,32,32))

                softmax_prob = nnf.softmax(pred, dim = 1)
                prob = softmax_prob[0][label]

                if prob < min_prob: ######treshold comes here
                    min_prob = prob ###??pointer
                    expanding_node = m
                    res = masked_img

            if expanding_node == -1:
                with open('mask_loc.txt', 'a') as f:
                    print((mask_len * 2) % h, file=f)
                return res
            elif expanding_node == 0:
                y1 = np.clip(y1, 0, y1)
                y2 = np.clip(y1 + mask_len, 0, y2)
                y3 = np.clip(y2, 0, y2)
                x1 = np.clip(x1, 0, x1)
                x2 = np.clip(x1+mask_len, 0, x2)
                x3 = np.clip(x2, 0, x2)
            elif expanding_node == 1:
                y1 = np.clip(y1, 0, y1)
                y2 = np.clip(y1 + mask_len, 0, y2)
                y3 = np.clip(y2, 0, y2)
                x1 = np.clip(x2, 0, x2)
                x2 = np.clip(x2+mask_len, 0, x3)
                x3 = np.clip(x3, 0, x3)
            elif expanding_node == 2:
                y1 = np.clip(y2, 0, y2)
                y2 = np.clip(y2 + mask_len, 0, y3)
                y3 = np.clip(y3, 0, y3)
                x1 = np.clip(x1, 0, x1)
                x2 = np.clip(x1+mask_len, 0, x2)
                x3 = np.clip(x2, 0, x2)
            elif expanding_node == 3:
                y1 = np.clip(y2, 0, y2)
                y2 = np.clip(y2 + mask_len, 0, y3)
                y3 = np.clip(y3, 0, y3)
                x1 = np.clip(x2, 0, x2)
                x2 = np.clip(x2+mask_len, 0, x3)
                x3 = np.clip(x3, 0, x3)

            mask_len = mask_len//2
