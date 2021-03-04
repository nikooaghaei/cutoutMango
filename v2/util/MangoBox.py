from sys import path
import torch
from torch._C import dtype
import torch.nn.functional as nnf
import numpy as np
from util.tree import Tree

from torch.utils.data import Dataset

from PIL import Image
import os
import tqdm
import pickle
from time import time
from pathlib import Path

def load_from(path):
    with open(path, "rb") as fp:
        return pickle.load(fp)

class Mango(object):
    """Mask out most important part of an image.

    Args:
        model (torch.model): Pytorch model to mask according to
        data (list - [(img, lbl), ...]): Image data to mask
        folder_name (str): Name of the folder to save the images on
        n_masks (int): Number of masks to cut out of each image. -ongoing
    """
    def __init__(self, model, data, 
                 folder_name = str(time()), 
                 n_masks = 4):
        self.model = model
        self.n_masks = n_masks
        self.data = data
        
        self.root = None # ??
        self.root_label = None # ??
        # self.treshold = 1
        
        self.folder_name = folder_name
        self.path = "data/MANGO/" + folder_name

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def _run_masking(self, root_img):
        """
        Args:
            root_img (Tensor): Tensor image of size (C, H, W).
        Returns:
            TODO
        """
        self.root = Tree(root_img)
    
        with torch.no_grad(): 
            image = self.root.data.view(1,3,32,32).to(self.device)
            pred = self.model(image).to(self.device)
    
        value, index = nnf.softmax(pred, dim = 1).max(1)
        self.root.prob = value
        self.root_label = index[0]

        res = self._make_tree(self.root)

        return res.data, res.mask_loc

    def _make_tree(self, node):
        """
        TODO: Fix the argument and returns
        Builds a tree from root node and assign the most important 
        node to self.res - Returns nothing
        """
        h = node.data.size(1)
        w = node.data.size(2)

        mask_color = 0 # 0 - Black
        masks = np.ones((self.n_masks, h, w), np.float32)

        if not node.mask_loc:
            y1 = np.clip(0, 0, h)
            y2 = np.clip(h // 2, 0, h)
            y3 = np.clip(h, 0, h)
            x1 = np.clip(0, 0, w)
            x2 = np.clip(w // 2, 0, w)
            x3 = np.clip(w, 0, w)
        else:
            y1 = np.clip(node.mask_loc[0], 0, node.mask_loc[0])
            y2 = np.clip((node.mask_loc[0] + node.mask_loc[1]) // 2, 
                          0, node.mask_loc[1])
            y3 = np.clip(node.mask_loc[1], 0, node.mask_loc[1])
            x1 = np.clip(node.mask_loc[2], 0, node.mask_loc[2])
            x2 = np.clip((node.mask_loc[2] + node.mask_loc[3]) // 2, 
                          0, node.mask_loc[3])
            x3 = np.clip(node.mask_loc[3], 0, node.mask_loc[3])

        y = [y1,y2,y3]
        x = [x1,x2,x3]
       
        masks = torch.from_numpy(masks)
        
        for mask in masks:
            mask = mask.expand_as(self.root.data)
        # masks = masks.cuda() # ??

        temp = 0 # mask number while creating it
        for j in range(len(y) - 1):
            for k in range(len(x) - 1):
                masks[temp][y[j]: y[j+1], x[k]: x[k+1]] = mask_color

                # building child image data
                child_data = self.root.data * masks[temp]

                # building child mask coordinates 
                # next sol: assign one of 4 corners to each mask so with depth 
                # u have the size and with corner (and parents' corners) you'll
                # have the loc BUT: order of tree length
                mask_coord = (y[j], y[j+1], x[k], x[k+1])

                # building child probability
                with torch.no_grad():
                    img = child_data.view(1,3,32,32).to(self.device)
                    pred = self.model(img).to(self.device)
                softmax_prob = nnf.softmax(pred, dim = 1)

                # adding new child to tree
                node.add_child(Tree(child_data, 
                                    softmax_prob[0][self.root_label], 
                                    mask_coord))

                temp += 1

        if(node.children[0].prob < node.prob):
            return node.children[0]
        return node

    def _show_chain(self, folder_name):
        '''
        TODO
        '''
        node = self.root
        parent_num = 0 # parent0 == root
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

    def create_dataset(self):
        '''
        TODO Add, Args - returns - description
        '''
        print("Beginnnig data processing...")
        self.model.eval()
        masked_data = [] 
        masked_labels = []

        for (images, labels) in tqdm.tqdm(self.data):
            # for each batch
            # [(img_1, lbl_1), (img_2, lbl_2), ...]
            for img, lbl in zip(images, labels):
                with torch.no_grad():
                    masked_img, if_masked = self._run_masking(img)
                    if if_masked:
                        masked_data.append(masked_img)
                        masked_labels.append(lbl)

        maskD = maskedDataset(masked_data, masked_labels)
        # create data/MANGO folder if needed
        print(self.path, "created...")
        Path(self.path).mkdir(parents=True, exist_ok=True)
        with open(self.path + '/maskD.txt', "wb") as fp:
            pickle.dump(maskD, fp)

        print("Masked data has been successfuly created.")

        return maskD

class maskedDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.targets = labels

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        TODO: Add transform - TIP VisionDataset ?
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        # img = Image.fromarray(img)

        # if self.transform is not None:
        #     img = self.transform(img)

        # if self.target_transform is not None:
        #     target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        return len(self.data)