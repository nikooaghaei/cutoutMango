import torch
import numpy as np
import copy
import torch.nn.functional as nnf
from util.mask_color import maskByColorDistance

import random
import time
import cProfile
from torchvision.utils import save_image

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
                temp_img = temp_img.view(1, 3, 32, 32).to(self.device)
                pred = self.model(temp_img).to(self.device)
                if min_prob == -1:  # initial prediction
                    value, index = nnf.softmax(pred, dim=1).max(1)
                    min_prob = value
                    label = index[0]
                    continue
                softmax_prob = nnf.softmax(pred, dim=1)
                prob = softmax_prob[0][label]
                if prob < min_prob:
                    min_prob = prob
                    res_img = temp_img
        return res_img.view(3, 32, 32).to(self.device)

class OrigMANGO_CUT(object):
    def __init__(self, model, args, data_size=50000):
        self.model = model
        self.n_branches = args.mng_n_branches
        self.mask_len = args.mng_init_len
        self.device = args.device
        self.n_calls = 0  # used as index of images
        # self.branches = {}

    def __call__(self, img):
        # img_id = get_id(img)
        
        h = img.size(1)
        w = img.size(2)
        # if img_id not in self.branches:  # still in first epoch
        # change to decide how deep to go. E.g. to go down to 1 pixel mask size set it to 1
        temp_mask_len = self.mask_len
        min_mask_len = 7
        y1 = 0
        y2 = self.mask_len
        y3 = h
        x1 = 0
        x2 = self.mask_len
        x3 = w
        with torch.no_grad(): 
            root_image = img.view(1,3,32,32).to(self.device)
            try:
                pred = self.model(root_image).to(self.device)
            except:
                pred = self.model(root_image)
        value, index = nnf.softmax(pred, dim = 1).max(1)
        ###############new
        # value = nnf.softmax(pred, dim = 1)[0][label]
        ##################
        min_prob = value
        label = index[0]
        branch = -2 # referring to original img
        while temp_mask_len > min_mask_len:
            masks = np.ones((self.n_branches, h, w), np.float32)
            
            masks[0][y1: y2, x1: x2] = 0. # or any other colors for mask
            masks[1][y1: y2, x2: x3] = 0.
            masks[2][y2: y3, x1: x2] = 0.
            masks[3][y2: y3, x2: x3] = 0.
            masks = torch.from_numpy(masks)
            
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
            if branch < 0:  #root or parent from previous level is the answer
                break
            temp_mask_len = temp_mask_len//2
            if branch == 0:
                # self.branches[img_id] = (y1, y2, x1, x2)
                by1,by2,bx1,bx2 = (y1, y2, x1, x2)
                #### creating mng mask for next level ####
                y3 = y2
                y2 = np.clip(y1 + temp_mask_len, 0, y2)
                x3 = x2
                x2 = np.clip(x1+temp_mask_len, 0, x2)
            elif branch == 1:
                by1,by2,bx1,bx2 = (y1, y2, x2, x3)
                #### creating mng mask for next level ####
                y3 = y2
                y2 = np.clip(y1 + temp_mask_len, 0, y2)
                x1 = x2
                x2 = np.clip(x2+temp_mask_len, 0, x3)
            elif branch == 2:
                by1,by2,bx1,bx2 = (y2, y3, x1, x2)
                #### creating mng mask for next level ####
                y1 = y2
                y2 = np.clip(y2 + temp_mask_len, 0, y3)
                x3 = x2
                x2 = np.clip(x1+temp_mask_len, 0, x2)
            elif branch == 3:
                by1,by2,bx1,bx2 = (y2, y3, x2, x3)
                #### creating mng mask for next level ####
                y1 = y2
                y2 = np.clip(y2 + temp_mask_len, 0, y3)
                x1 = x2
                x2 = np.clip(x2+temp_mask_len, 0, x3)
            branch = -1 #refers to previous parent which is not root (root => -2)
    
        if branch == -2:
            # self.branches[img_id] = (-2, -2, -2, -2)
            by1,by2,bx1,bx2 = (-2, -2, -2, -2)

        # self.branches[img_id] = by1,by2,bx1,bx2
        # else:
            # by1,by2,bx1,bx2 = self.branches[img_id] #border coordinates for loacting mask center
        
        ###########test
        # self.n_calls = self.n_calls + 1
        # if self.n_calls % 50000 == 0:
        #     print(self.n_calls)
        # elif (self.n_calls < 50000 and len(self.branches) != self.n_calls) or (self.n_calls > 50000 and len(self.branches) == self.n_calls):
        #     print('HEREeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee')
        #     print(self.n_calls, len(self.branches))
        #     self.n_calls = self.n_calls - 1
        #############
        
        if (by1,by2,bx1,bx2) == (-2,-2,-2,-2):
            return img
        ####creating rand mask####
        yc = np.random.randint(by1, by2)    #mask center coordinates
        xc = np.random.randint(bx1, bx2)
        yc1 = np.clip(yc - self.mask_len // 2, 0, h)
        yc2 = np.clip(yc + self.mask_len // 2, 0, h)
        xc1 = np.clip(xc - self.mask_len // 2, 0, w)
        xc2 = np.clip(xc + self.mask_len // 2, 0, w)

        rmask = np.ones((h, w), np.float32)
        rmask[yc1: yc2, xc1: xc2] = 0.
        rmask = torch.from_numpy(rmask)
        rmask = rmask.expand_as(img)
        img = img * rmask
        return img

class MANGO_CUT_S(object):
    """If any of 1/4 of the root is chosen by MANGo, it applies Cutout on that quarter.
        If the root is chosen by MANGO, then no Cutout is applied root is returned.
    Args:
        model (torch.model): Pytorch model to mask according to
        data (list - [(img, lbl), ...]): Image data to mask
        folder_name (str): Name of the folder to save the images on
        n_branches (int): Each node is divided to n_branches nodes -ongoing
    """

    def __init__(self, model, args, data_size=50000):
        self.model = model
        self.n_branches = args.mng_n_branches
        self.init_length = args.mng_init_len
        self.device = args.device

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with one hole of dimension 
            ((mask_len * 2) % h x (mask_len * 2) % h) cut out of it.
            Integer (mask_len * 2) % h : Length of the mask cut out of the image.
        """
        h = img.size(1)
        w = img.size(2)
        y1 = 0
        y2 = self.init_length
        y3 = h
        x1 = 0
        x2 = self.init_length
        x3 = w
        
        with torch.no_grad():
            root_image = img.view(1, 3, 32, 32).to(self.device)
            try:
                pred = self.model(root_image).to(self.device)
            except:
                pred = self.model(root_image)
        value, index = nnf.softmax(pred, dim=1).max(1)
        ###############
        # value = nnf.softmax(pred, dim = 1)[0][label]
        ##################
        min_prob = value
        label = index[0]
        masks = np.ones((self.n_branches, h, w), np.float32)

        masks[0][y1: y2, x1: x2] = 0.  # or any other colors for mask
        masks[1][y1: y2, x2: x3] = 0.
        masks[2][y2: y3, x1: x2] = 0.
        masks[3][y2: y3, x2: x3] = 0.
        masks = torch.from_numpy(masks)
        branch = -1  # referring to original img

        for m, mask in enumerate(masks):
            mask = mask.expand_as(img)
            masked_img = img * mask
            # building child probability
            with torch.no_grad():
                temp_img = masked_img.view(1, 3, 32, 32).to(self.device)
                try:
                    pred = self.model(temp_img).to(self.device)
                except:
                    pred = self.model(temp_img)
            softmax_prob = nnf.softmax(pred, dim=1)
            prob = softmax_prob[0][label]

            if prob < min_prob:  # treshold comes here
                min_prob = prob
                branch = m

        if branch == -1:  # no mask- original image
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
        return img

class MANGO_CUT(object):
    """If any of 1/4 of the root is chosen by MANGo, it applies Cutout on that quarter.
        If the root is chosen by MANGO, then no Cutout is applied root is returned.
    Args:
        model (torch.model): Pytorch model to mask according to
        data (list - [(img, lbl), ...]): Image data to mask
        folder_name (str): Name of the folder to save the images on
        n_branches (int): Each node is divided to n_branches nodes -ongoing
    """

    def __init__(self, model, args, data_size=50000):
        self.model = model
        self.n_branches = args.mng_n_branches
        self.init_length = args.mng_init_len
        self.device = args.device
        self.data_size = data_size
        self.n_calls = 0  # used as index of images
        self.branches = {}
        # self.branches =[]

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with one hole of dimension 
            ((mask_len * 2) % h x (mask_len * 2) % h) cut out of it.
            Integer (mask_len * 2) % h : Length of the mask cut out of the image.
        """
        # np.set_printoptions(threshold=np.inf)
        # t = time.time()
        # img_id = str(img)
        # img_id = np.array2string(img.numpy(), threshold=np.inf, suppress_small=False)
        img_id = get_id(img)
        # img_id = np.array_str(img.numpy(), suppress_small=False)
        # print(time.time() - t)
        h = img.size(1)
        w = img.size(2)
        y1 = 0
        y2 = self.init_length
        y3 = h
        x1 = 0
        x2 = self.init_length
        x3 = w
        # if self.n_calls >= self.data_size:
        #     for key in self.branches:
        #         if key is img_id:
        #             branch = self.branches[img_id]
        #             break
        if img_id not in self.branches:  # still in first epoch
            # if img_id not in self.branches: # still in first epoch
            # if self.n_calls < self.data_size:
            with torch.no_grad():
                root_image = img.view(1, 3, 32, 32).to(self.device)
                try:
                    pred = self.model(root_image).to(self.device)
                except:
                    pred = self.model(root_image)
            value, index = nnf.softmax(pred, dim=1).max(1)
            ###############
            # value = nnf.softmax(pred, dim = 1)[0][label]
            ##################
            min_prob = value
            label = index[0]
            masks = np.ones((self.n_branches, h, w), np.float32)

            masks[0][y1: y2, x1: x2] = 0.  # or any other colors for mask
            masks[1][y1: y2, x2: x3] = 0.
            masks[2][y2: y3, x1: x2] = 0.
            masks[3][y2: y3, x2: x3] = 0.
            masks = torch.from_numpy(masks)
            branch = -1  # referring to original img

            for m, mask in enumerate(masks):
                mask = mask.expand_as(img)
                masked_img = img * mask
                # building child probability
                with torch.no_grad():
                    temp_img = masked_img.view(1, 3, 32, 32).to(self.device)
                    try:
                        pred = self.model(temp_img).to(self.device)
                    except:
                        pred = self.model(temp_img)
                softmax_prob = nnf.softmax(pred, dim=1)
                prob = softmax_prob[0][label]

                if prob < min_prob:  # treshold comes here
                    min_prob = prob
                    branch = m

            self.branches[img_id] = branch
            # self.branches.append(branch)
        else:
            branch = self.branches[img_id]
            # branch = self.branches[self.n_calls % self.data_size]

        self.n_calls = self.n_calls + 1
        if branch == -1:  # no mask- original image
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

        if len(self.branches) != self.n_calls:
            print('HERE', len(self.branches), self.n_calls)
            self.n_calls -= 1
            exit()
        return img


class MANGOCut_MaskDiff(object):
    """If any of 1/4 of the root is chosen by MANGo, it applies Cutout on that quarter.
        If the root is chosen by MANGO, then no Cutout is applied root is returned.
    Args:
        model (torch.model): Pytorch model to mask according to
        data (list - [(img, lbl), ...]): Image data to mask
        folder_name (str): Name of the folder to save the images on
        n_branches (int): Each node is divided to n_branches nodes -ongoing
    """

    def __init__(self, model, args, data_size=50000):
        self.model = model
        self.n_branches = args.mng_n_branches
        self.mask_len = args.mng_init_len
        self.device = args.device

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with one hole of dimension 
            ((mask_len * 2) % h x (mask_len * 2) % h) cut out of it.
            Integer (mask_len * 2) % h : Length of the mask cut out of the image.
        """
        # img_id = np.array2string(img.numpy())
        h = img.size(1)
        w = img.size(2)
        # change to decide how deep to go. E.g. to go down to 1 pixel mask size set it to 1
        temp_mask_len = self.mask_len
        min_mask_len = 7
        y1 = 0
        y2 = self.mask_len
        y3 = h
        x1 = 0
        x2 = self.mask_len
        x3 = w
        with torch.no_grad(): 
            root_image = img.view(1,3,32,32).to(self.device)
            try:
                pred = self.model(root_image).to(self.device)
            except:
                pred = self.model(root_image)
        value, index = nnf.softmax(pred, dim = 1).max(1)
        ###############new
        # value = nnf.softmax(pred, dim = 1)[0][label]
        ##################
        min_prob = value
        label = index[0]
        branch = -2 # referring to original img
        while temp_mask_len > min_mask_len:
            masked_imgs = torch.stack([
            maskByColorDistance(img, [y1, y2, x1, x2]),
            maskByColorDistance(img, [y1, y2, x2, x3]),
            maskByColorDistance(img, [y2, y3, x1, x2]),
            maskByColorDistance(img, [y2, y3, x2, x3])])
            
            for m, masked_img in enumerate(masked_imgs):
                with torch.no_grad():
                    temp_img = masked_img.view(1, 3, 32, 32).to(self.device)
                    try:
                        pred = self.model(temp_img).to(self.device)
                    except:
                        pred = self.model(temp_img)
                softmax_prob = nnf.softmax(pred, dim=1)
                prob = softmax_prob[0][label]

                if prob < min_prob:  # treshold comes here
                    min_prob = prob
                    branch = m
            if branch < 0:  #root or parent from previous level is the answer
                break
            temp_mask_len = temp_mask_len//2
            if branch == 0:
                by1,by2,bx1,bx2 = (y1, y2, x1, x2)  #border coordinates so far
                #### creating mng mask for next level ####
                y3 = y2
                y2 = np.clip(y1 + temp_mask_len, 0, y2)
                x3 = x2
                x2 = np.clip(x1+temp_mask_len, 0, x2)
            elif branch == 1:
                by1,by2,bx1,bx2 = (y1, y2, x2, x3)
                #### creating mng mask for next level ####
                y3 = y2
                y2 = np.clip(y1 + temp_mask_len, 0, y2)
                x1 = x2
                x2 = np.clip(x2+temp_mask_len, 0, x3)
            elif branch == 2:
                by1,by2,bx1,bx2 = (y2, y3, x1, x2)
                #### creating mng mask for next level ####
                y1 = y2
                y2 = np.clip(y2 + temp_mask_len, 0, y3)
                x3 = x2
                x2 = np.clip(x1+temp_mask_len, 0, x2)
            elif branch == 3:
                by1,by2,bx1,bx2 = (y2, y3, x2, x3)
                #### creating mng mask for next level ####
                y1 = y2
                y2 = np.clip(y2 + temp_mask_len, 0, y3)
                x1 = x2
                x2 = np.clip(x2+temp_mask_len, 0, x3)
            branch = -1 #refers to previous parent which is not root (root => -2)
    
        if branch == -2:
            return img
        ####creating rand mask####
        yc = np.random.randint(by1, by2)    #mask center coordinates
        xc = np.random.randint(bx1, bx2)
        yc1 = np.clip(yc - self.mask_len // 2, 0, h)
        yc2 = np.clip(yc + self.mask_len // 2, 0, h)
        xc1 = np.clip(xc - self.mask_len // 2, 0, w)
        xc2 = np.clip(xc + self.mask_len // 2, 0, w)
        res =  maskByColorDistance(img, [yc1, yc2, xc1, xc2])
        if yc2-yc1 == 8 and xc2-xc1 == 8 and \
            not ((yc == 0 and xc==0) or (yc == 0 and xc==32) \
            or (yc == 32 and xc==0) or (yc == 32 and xc==32)):
            save_image(res, 'new/'+str(random.random())+'.png')
        return res

def get_id(a):
    '''
    convert np array to string var
    '''
    s = ""
    for width in a:
        for height in width:
            for channel in height:
                s += str(channel.item())
    return s
