import torch
import torch.nn.functional as nnf
import numpy as np
# from tree import Tree   #####should be changed to util.tree
from util.tree import Tree

import matplotlib.pyplot as plt
from PIL import Image
from torchvision.utils import save_image
import torchvision

import os
import sys
import faulthandler
              

class Mango(object):
    """Randomly mask out one or more patches from an image.

    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, model, n_masks = 4):
        self.model = model
        self.n_masks = n_masks
        #self.treshold = 1 ##########################TO BE SET??????
        self.root = None
        self.root_label = None
        self.res = None #variable to keep main node of root's tree (by calling make_tree func)

        self.test = 0
    def __call__(self, root_img):    ######not sure if we should have __call?
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
#        torch.set_printoptions(profile="full")
        
        self.root = Tree(root_img)
    
        with torch.no_grad(): 
            pred = self.model(self.root.data.view(1,3,32,32))
    
        value, index = nnf.softmax(pred, dim = 1).max(1)
        self.root.prob = value
        self.root_label = index[0]

        ####temporary testing if prob for some classes can be equal##to be removed
   #     temp = nnf.softmax(pred, dim = 1)
  #      for i in range(temp.size(1)):
 #           if(self.root.prob == temp[0][i] and index[0] != i):
#                print("*")
            
        self.make_tree(self.root)
       
        ####to be removed
#        print(self.res.mask_loc)

        return self.res.data

        #######for returning the main masked part
        # #################not good - It's just for 4 masks
        # if self.res.mask_loc:      ##if final main node is not the root
        #     main_masked_img = self.crop_img(self.root.data, self.res.mask_loc[0], self.res.mask_loc[1], self.res.mask_loc[2], self.res.mask_loc[3]) #return self.root_img - self.res.data=masked part
        #     return main_masked_img
        # else:   #main node was main root image with no masking
        #     return self.root.data

    def shp(self, t):   #t = tensor
        return t.view(1, t.shape[0], t.shape[1])

    def crop_img(self, img, y_b, y_e, x_b, x_e):
        squares = []
        for ch in range(img.size(0)):  #img.size(0) = num of channels in the img
            squares.append(img[ch][y_b:y_e,x_b:x_e])
        return torch.cat([self.shp(squares[0]), self.shp(squares[1]), self.shp(squares[2])], dim=0)   #baaaaadd-just for 3 channel img

    def make_tree(self, node):
        """
        Builds a tree from root node and assign the most important node to self.res - Returns nothing##########how to write comment
        """
        h = self.root.data.size(1)
        w = self.root.data.size(2)

        masks = np.ones((self.n_masks, h, w), np.float32)

        ####not good - It's just for 4 masks but easy to generalize it
        if not node.mask_loc:   #if node == self.root
            y1 = np.clip(0, 0, h)
            y2 = np.clip(h // 2, 0, h)
            y3 = np.clip(h, 0, h)
            x1 = np.clip(0, 0, w)
            x2 = np.clip(w // 2, 0, w)
            x3 = np.clip(w, 0, w)

        else:
            ####not good - It's just for 4 masks but easy to generalize it
            y1 = np.clip(node.mask_loc[0], 0, node.mask_loc[0])
            y2 = np.clip((node.mask_loc[0] + node.mask_loc[1]) // 2, 0, node.mask_loc[1]) #node.mask_loc[1] - (node.mask_loc[1] - node.mask_loc[0]) //2
            y3 = np.clip(node.mask_loc[1], 0, node.mask_loc[1])
            x1 = np.clip(node.mask_loc[2], 0, node.mask_loc[2])
            x2 = np.clip((node.mask_loc[2] + node.mask_loc[3]) // 2, 0, node.mask_loc[3])
            x3 = np.clip(node.mask_loc[3], 0, node.mask_loc[3])

        y = [y1,y2,y3]
        x = [x1,x2,x3]

        # masks[0][y1: y2, x1: x2] = 0.   ########or any other colors for mask
        # masks[1][y1: y2, x2: x3] = 0.
        # masks[2][y2: y3, x1: x2] = 0.
        # masks[3][y2: y3, x2: x3] = 0.
       
        masks = torch.from_numpy(masks)
        
        for mask in masks:
            mask = mask.expand_as(self.root.data)
        masks = masks.cuda()

        temp = 0    #mask number while creating it
        for j in range(len(y) - 1):
            for k in range(len(x) - 1):
                masks[temp][y[j]: y[j+1], x[k]: x[k+1]] = 0  ########or any other colors for mask

                ###building child image data
                child_data = self.root.data * masks[temp]

                ####building child mask coordinates ############next sol:assign one of 4 corners to each mask so with depth u have the size and with corner (and parents' corners) you'll have the loc BUTT: order of tree length
                mask_coord = (y[j], y[j+1], x[k], x[k+1])

                #####building child probability
                with torch.no_grad():
                    pred = self.model(child_data.view(1,3,32,32))
                softmax_prob = nnf.softmax(pred, dim = 1)

                #####adding new child to tree
                node.add_child(Tree(child_data, softmax_prob[0][self.root_label], mask_coord))

                temp += 1

#        treshold = 1 + ((node.children[0].mask_loc[1] - node.children[0].mask_loc[0]) / (self.n_masks * h))  ##1+(mask_size/(img_size * n_masks))

 #       if(node.children[0].mask_loc[1] > node.children[0].mask_loc[0] + 1 and node.children[0].prob < treshold * node.prob):    ####determining to what depth we go down => here is down to mask size = 1 pixel)
                ###as far as children are ordered, we only expand first if the condition is true
                ###############later changes if you need more than one branch expansion
  #          self.make_tree(node.children[0])
   #     else:
#        treshold = 1
        if(node.children[0].prob < node.prob):  ##main is one pixel size mask
            self.res = node.children[0]
        else:
            self.res = node
        return

    def show_chain(self, folder_name):
        node = self.root
        parent_num = 0  #parent0 == root
        while(node.children):
            if not os.path.exists(folder_name):
                os.makedirs(folder_name)

            save_image(node.data, folder_name + '/parent' + str(parent_num) + '.png')   #parent0 is the root
            
            child_num = 1
            for child in node.children:
                save_image(child.data, folder_name + '/child' + str(parent_num) + str(child_num) + '.png')
                child_num = child_num + 1
            
            node = node.children[0]
            parent_num = parent_num + 1
            
