import torch
import numpy as np
import copy
import torch.nn.functional as nnf

import time
import cProfile
class Cutout_FIXED(object):
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

class Cutout_NOTRANSF(object):
    """original cutout being applied but not as a data transform!
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(1):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img

class MANGO_FIXED(object):
    def __init__(self, model, args):
        self.model = model
        self.n_holes = args.cutout_n_holes
        self.length = args.cutout_len
        self.device = args.device

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
                if prob < min_prob:
                    min_prob = prob
                    res_img = temp_img
        return res_img.view(3,32,32).to(self.device)

class MANGO_CUT(object):
    """If any of 1/4 of the root is chosen by MANGo, it applies Cutout on that quarter.
        If the root is chosen by MANGO, then no Cutout is applied root is returned.
    Args:
        model (torch.model): Pytorch model to mask according to
        data (list - [(img, lbl), ...]): Image data to mask
        folder_name (str): Name of the folder to save the images on
        n_branches (int): Each node is divided to n_branches nodes -ongoing
    """
    def __init__(self, model, args, data_size = 50000):
        self.model = model
        self.n_branches = args.mng_n_branches
        self.init_length = args.mng_init_len
        self.device = args.device
        self.data_size = data_size
        self.n_calls = 0    #used as index of images
        self.probs = []

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with one hole of dimension 
            ((mask_len * 2) % h x (mask_len * 2) % h) cut out of it.
            Integer (mask_len * 2) % h : Length of the mask cut out of the image.
        """
        # timing=[]
        # a = time.time()

        h = img.size(1)
        w = img.size(2)
        # b = time.time()
        # timing.append(('h/w size', b-a))
        y1 = 0
        y2 = self.init_length
        y3 = h
        x1 = 0
        x2 = self.init_length
        x3 = 0
        # timing.append(('before if/else', time.time() - a)) 
        # print(self.n_calls)
        if self.n_calls < self.data_size: # still in first epoch

            # a = time.time()

            with torch.no_grad(): 
                root_image = img.view(1,3,32,32).to(self.device)
                try:
                    pred = self.model(root_image).to(self.device)
                except:
                    pred = self.model(root_image)
            value, index = nnf.softmax(pred, dim = 1).max(1)
            ###############
            # value = nnf.softmax(pred, dim = 1)[0][label]
            ##################
            min_prob = value
            label = index[0]
            masks = np.ones((self.n_branches, h, w), np.float32)
            
            masks[0][y1: y2, x1: x2] = 0. # or any other colors for mask
            masks[1][y1: y2, x2: x3] = 0.
            masks[2][y2: y3, x1: x2] = 0.
            masks[3][y2: y3, x2: x3] = 0.
            masks = torch.from_numpy(masks)
            branch = -1 # referring to original img
            
            for m, mask in enumerate(masks):
                mask = mask.expand_as(img)
                masked_img = img * mask #.to(self.device) # Why error
                #####building child probability
                with torch.no_grad():
                    temp_img = masked_img.view(1,3,32,32).to(self.device)
                    try:
                        pred = self.model(temp_img).to(self.device)
                    except:
                        pred = self.model(temp_img)
                softmax_prob = nnf.softmax(pred, dim = 1)
                prob = softmax_prob[0][label]

                if prob < min_prob: ######treshold comes here
                    min_prob = prob ###??pointer
                    branch = m
            
            self.probs.append(branch)
            # timing.append(('inside if', time.time() -a ))

        else:

            # a = time.time()

            branch = self.probs[self.n_calls % self.data_size]
            # timing.append(('inside else', time.time() -a ))

        # a = time.time()

        self.n_calls = self.n_calls + 1
        if branch == -1:    #no mask- original image
            return img

        elif branch == 0:
            y = np.random.randint(self.init_length)
            x = np.random.randint(self.init_length)

        elif branch == 1:
            y = np.random.randint(self.init_length)
            x = np.random.randint(self.init_length, w)

        elif branch == 2:
            y = np.random.randint(self.init_length, h)
            x = np.random.randint(self.init_length)

        elif branch == 3:
            y = np.random.randint(self.init_length, h)
            x = np.random.randint(self.init_length, w)

        y1 = np.clip(y - self.init_length // 2, 0, h)
        y2 = np.clip(y + self.init_length // 2, 0, h)
        x1 = np.clip(x - self.init_length // 2, 0, w)
        x2 = np.clip(x + self.init_length // 2, 0, w)
        
        mask = np.ones((h, w), np.float32)
        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        # timing.append(('before return', time.time() -a))
        # sumt = 0
        # for txt, t in timing:
        #     sumt += t    
        # for txt, t in timing:
        #     print(txt, ':', t*100/sumt)
        return img