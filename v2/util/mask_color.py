import torch
import numpy as np
import copy
from torchvision.utils import save_image
import random
def maskByColorDistance(img, mask_loc):
    # get the max value img has
    # get the min value img has
    # find the avg
    # if value is below avg make it equal to min val
    # if value is above avg make it equal to max val
    # print(img.shape, mask_loc)
    cp_img = copy.deepcopy(img)
    masked_img = cp_img[:, mask_loc[0]:mask_loc[1], mask_loc[2]:mask_loc[3]] # c, w, h

    for i, channel in enumerate(cp_img):
        c_max_val = torch.max(channel)
        c_min_val = torch.min(channel)
        c_avg = (c_max_val + c_min_val) / 2
        masked_img[i][masked_img[i] < c_avg] = c_max_val + 1
        masked_img[i][masked_img[i] != c_max_val + 1] = c_min_val
        masked_img[i][masked_img[i] == c_max_val + 1] = c_max_val
    
    
    ####another approach ####
    # max_val = torch.max(cp_img)
    # min_val = torch.min(cp_img)
    # avg = (max_val - min_val) / 2
    # masked_img[masked_img > avg] = masked_img[masked_img > avg] - avg
    # masked_img[masked_img < avg] = masked_img[masked_img < avg] + avg
    
    cp_img[:, mask_loc[0]:mask_loc[1], mask_loc[2]:mask_loc[3]] = masked_img
    return cp_img

# print(colorByDistance(torch.tensor([-0.5, 0, 0.3 ,0.5])))
# img = torch.randn(32,32,3)

# mask_imgs = np.array([])
# mask_imgs = maskByColorDistance(img, [16, 32, 16, 32])
# # np.append(mask_imgs, )
# print(mask_imgs[16:32, 16:32, :])