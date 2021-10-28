import torch
import numpy as np
from torchvision.utils import save_image

def rand_pix(img, mask_loc, mask = None):
    cp_img = img.detach().clone()
    if mask is None:
        mask = cp_img[:, mask_loc[0]:mask_loc[1], mask_loc[2]:mask_loc[3]] # c, w, h

        ch_n_255 = []
        ch_n_0 = []
        for m, sd in zip([125.3, 123.0, 113.9],[63.0, 62.1, 66.7]):
            ch_n_255.append((255 - m) / sd)
            ch_n_0.append((0 - m) / sd)    
        ####RANDOM#####
        for channel in range(mask.size(0)):
            sh = mask[channel].size()
            mask[channel] = torch.tensor(np.random.uniform(ch_n_0[channel], ch_n_255[channel], size=sh))
        
    cp_img[:, mask_loc[0]:mask_loc[1], mask_loc[2]:mask_loc[3]] = mask
    # save_image(cp_img,'test/'+str(np.random.random())+'.png')
    return cp_img
def far_edge(img, mask_loc, mask = None):
    # get the max value img has
    # get the min value img has
    # find the avg
    # if value is below avg make it equal to min val
    # if value is above avg make it equal to max val
    # print(img.shape, mask_loc)
    cp_img = img.detach().clone()
    if mask is None:
        mask = cp_img[:, mask_loc[0]:mask_loc[1], mask_loc[2]:mask_loc[3]] # c, w, h
  
        ####FAR EDGE####
        for channel in cp_img:
            c_max_val = torch.max(channel)
            c_min_val = torch.min(channel)
            c_avg = (c_max_val + c_min_val) / 2
            mask[mask < c_avg] = c_max_val + 1
            mask[mask != c_max_val + 1] = c_min_val
            mask[mask == c_max_val + 1] = c_max_val

    cp_img[:, mask_loc[0]:mask_loc[1], mask_loc[2]:mask_loc[3]] = mask
    # save_image(cp_img,'test/'+str(np.random.random())+'.png')
    return cp_img
# print(colorByDistance(torch.tensor([-0.5, 0, 0.3 ,0.5])))
# img = torch.randn(32,32,3)

# mask_imgs = np.array([])
# mask_imgs = maskByColorDistance(img, [16, 32, 16, 32])
# # np.append(mask_imgs, )
# print(mask_imgs[16:32, 16:32, :])