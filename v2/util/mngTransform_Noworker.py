import enum
from typing import Counter
from numpy.lib.type_check import imag
import torch
import numpy as np
import copy
import torch.nn.functional as nnf
from util.mask_color import rand_pix
from util.cutout import Cutout

import random
import time
import cProfile
from torchvision.utils import save_image

class OrigMANGO(torch.nn.Module):
    def __init__(self, model, args):
        super().__init__()
        self.model = model
        self.n_branches = args.mng_n_branches
        self.mask_len = args.mng_init_len
        self.device = args.device
        self.n_calls = 1
        self.imgs = torch.empty(args.batch_size,3,32,32)
        # self.imgs = []

    def __call__(self, img, true_labels = None):        
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with one hole of dimension 
            ((mask_len * 2) % h x (mask_len * 2) % h) cut out of it.
            Integer (mask_len * 2) % h : Length of the mask cut out of the image.
        """
        # print('mango calls:',self.n_calls)
        self.imgs[(self.n_calls%128) - 1] = img
        # self.imgs.append(img)
        self.n_calls = self.n_calls + 1
        if (self.n_calls - 1) % 128 != 0 and self.n_calls != 50001:
        # if len(self.imgs) % 128 != 0 and len(self.imgs) !=5000:
            return img

        if self.n_calls == 50001:
            imgs = self.imgs[0:80]
            self.n_calls = 1
        else:
            imgs = self.imgs
        # imgs = torch.stack(self.imgs)

        probability = np.random.randint(2)
        if probability == 0:           
            return imgs

        batch_size = imgs.size(0)
        c = imgs.size(1)
        h = imgs.size(2)
        w = imgs.size(3)
        temp_mask_len = self.mask_len
        # change to decide how deep to go. E.g. to go down to 1 pixel mask size set it to 1
        min_mask_len = (self.mask_len // 2) - 1
        y1 = np.zeros(batch_size, dtype=int)
        y2 = np.full(batch_size, np.clip(self.mask_len, 0, h), dtype=int)
        y3 = np.full(batch_size, h, dtype=int)
        x1 = np.zeros(batch_size, dtype=int)
        x2 = np.full(batch_size, np.clip(self.mask_len, 0, w), dtype=int)
        x3 = np.full(batch_size, w, dtype=int)
        self.model.eval()
        with torch.no_grad():
            try:
                preds = self.model(imgs.to(self.device)).to(self.device)
            except:
                preds = self.model(imgs.to(self.device)).to(self.device)
        # if true_labels: ####TOBE FIXED####
        #     value = nnf.softmax(preds, dim = 1)[0][true_labels]
        #     label = true_labels
        # else:
        min_probs, labels = nnf.softmax(preds, dim = 1).max(1)
        #########################################
        # min_probs = min_probs.fill_(1000.)
        #########################################
        # res = copy.deepcopy(imgs)
        res = imgs.detach().clone()
        # min_probs = torch.full((128,), -1.)
        # labels = torch.full((128,), 0)

        while temp_mask_len > min_mask_len:
            masks = np.ones((batch_size, self.n_branches, h, w), np.float32)
            for i in range(batch_size):
                masks[i][0][y1[i]: y2[i], x1[i]: x2[i]] = 0. # or any other colors for mask
                masks[i][1][y1[i]: y2[i], x2[i]: x3[i]] = 0.
                masks[i][2][y2[i]: y3[i], x1[i]: x2[i]] = 0.
                masks[i][3][y2[i]: y3[i], x2[i]: x3[i]] = 0.
            
            masked_imgs = np.ones((batch_size, self.n_branches, c, h, w), np.float32)
            masked_imgs = torch.from_numpy(masked_imgs)
            masks = torch.from_numpy(masks)
            
            for i in range(batch_size):
                for b in range(self.n_branches):
                    temp = masks[i][b].expand_as(imgs[i])
                    masked_imgs[i][b] = imgs[i] * temp

            branch = np.full(batch_size, -1) # referring to original img
            #####building child probability
            self.model.eval()
            with torch.no_grad():
                temp_imgs = masked_imgs.view(batch_size*self.n_branches, c, h, w).to(self.device)
                try:
                    preds = self.model(temp_imgs).to(self.device)
                except:
                    preds = self.model(temp_imgs)
            preds = preds.view(batch_size, self.n_branches, preds.size(1))
            
            softmax_probs = nnf.softmax(preds, dim = 2)

            temp_mask_len = temp_mask_len//2
            for i in range(batch_size):
                # if temp_mask_len < self.mask_len // 2 and branch[i] == -1:
                #     continue
                # softmax_probs = nnf.softmax(preds[i], dim = 1)
                probs = [softmax_probs[i][j][labels[i]] for j in range(self.n_branches)]
                for j in range(self.n_branches):
                    # prob = softmax_probs[i][j][labels[i]]
                    if probs[j] < min_probs[i]: ######treshold comes here
                        min_probs[i] = probs[j]
                        branch[i] = j
                        res[i] = masked_imgs[i][j]
                        # save_image(res[i], 'test/' + str(i)+str(j)+'.png')
                # if branch[i] == -1:
                #     continue
                if branch[i] == 0:
                    y3[i] = y2[i]
                    y2[i] = np.clip(y1[i] + temp_mask_len, 0, y2[i])
                    x3[i] = x2[i]
                    x2[i] = np.clip(x1[i]+temp_mask_len, 0, x2[i])
                elif branch[i] == 1:
                    y3[i] = y2[i]
                    y2[i] = np.clip(y1[i] + temp_mask_len, 0, y2[i])
                    x1[i] = x2[i]
                    x2[i] = np.clip(x2[i]+temp_mask_len, 0, x3[i])
                elif branch[i] == 2:
                    y1[i] = y2[i]
                    y2[i] = np.clip(y2[i] + temp_mask_len, 0, y3[i])
                    x3[i] = x2[i]
                    x2[i] = np.clip(x1[i]+temp_mask_len, 0, x2[i])
                elif branch[i] == 3:
                    y1[i] = y2[i]
                    y2[i] = np.clip(y2[i] + temp_mask_len, 0, y3[i])
                    x1[i] = x2[i]
                    x2[i] = np.clip(x2[i]+temp_mask_len, 0, x3[i])
        # for i in range(res.size(0)):
        #     save_image(res[i], 'test/final' + str(i)+'.png')
        return res
