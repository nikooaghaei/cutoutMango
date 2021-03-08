from sys import path
import torch
from torch._C import dtype
import torch.nn.functional as nnf
import numpy as np

from torch.utils.data import Dataset

from tqdm import tqdm
import pickle
from time import time
from pathlib import Path

def load_from(path):
    with open(path, "rb") as fp:
        return pickle.load(fp)

def run_mango(model, trainloader, length=16,
              load_from_path='',
              folder_name=str(time()),
              batch_size=16,
              n_masks=4,
              n_workers=2):
    if load_from_path:
        new_train = load_from(load_from_path)
        if not isinstance(new_train, maskedDataset):
            raise ValueError('Inappropriate type loaded: {} for the data \
                whereas a maskedDataset is expected'.format(type(new_train)))
    else:
        mango = Mango(model, trainloader, length, 
                      folder_name, n_masks)
        new_train = mango.create_dataset()
    
    return torch.utils.data.DataLoader(new_train, batch_size=batch_size,
                                       shuffle=True, num_workers=n_workers)

class Mango(object):
    """Mask out most important part of an image.

    Args:
        model (torch.model): Pytorch model to mask according to
        data (list - [(img, lbl), ...]): Image data to mask
        folder_name (str): Name of the folder to save the images on
        n_masks (int): Number of masks to cut out of each image. -ongoing
    """
    def __init__(self, model, data, length,
                 folder_name, n_masks):
        self.model = model
        self.n_masks = n_masks
        self.init_length = length
        self.data = data
        
        self.folder_name = folder_name
        self.path = "data/MANGO/" + folder_name

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def _run_masking(self, img):
        """
        Args:
            root_img (Tensor): Tensor image of size (C, H, W).
        Returns:
            TODO
        """
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
            root_image = img.view(1,3,32,32).to(self.device)
            try:
                pred = self.model(root_image).to(self.device)
            except:
                pred = self.model(root_image)
        
        value, index = nnf.softmax(pred, dim = 1).max(1)
        
        min_prob = value
        label = index[0]
        res = img
        while(mask_len > 0): # change to decide how deep to go
            masks = np.ones((self.n_masks, h, w), np.float32)
            
            masks[0][y1: y2, x1: x2] = 0. # or any other colors for mask
            masks[1][y1: y2, x2: x3] = 0.
            masks[2][y2: y3, x1: x2] = 0.
            masks[3][y2: y3, x2: x3] = 0.

            masks = torch.from_numpy(masks)

            expanding_node = -1 # referring to original img
            
            for m, mask in enumerate(masks):
                mask = mask.expand_as(img)
                masked_img = img * mask #.to(self.device) # Why error

                #####building child probability
                with torch.no_grad():
                    root_image = masked_img.view(1,3,32,32).to(self.device)
                    try:
                        pred = self.model(root_image).to(self.device)
                    except:
                        _, pred = self.model(root_image)

                softmax_prob = nnf.softmax(pred, dim = 1)
                prob = softmax_prob[0][label]

                if prob < min_prob: ######treshold comes here
                    min_prob = prob ###??pointer
                    expanding_node = m
                    res = masked_img

            if expanding_node == -1:
                # with open('mask_loc.txt', 'a') as f:
                #     print((mask_len * 2) % h, file=f)
                return res, (mask_len * 2) % h
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

    def create_dataset(self):
        '''
        TODO Add, Args - returns - description
        '''
        print("Beginnnig data processing...")
        self.model.eval()
        original_data = []
        original_label = []
        masked_data = [] 
        masked_labels = []

        for (images, labels) in tqdm(self.data):
            # for each batch
            # [(img_1, lbl_1), (img_2, lbl_2), ...]
            for img, lbl in zip(images, labels):
                original_data.append(img)   #TODO
                original_label.append(lbl)
                with torch.no_grad():
                    masked_img, mask = self._run_masking(img)
                   
                    Path('./data/masks/').mkdir(parents=True, exist_ok=True)
                    with open('./data/masks/mask_loc.txt', 'a') as f:
                        print(mask, file=f)
                    
                    if mask:
                        masked_data.append(masked_img)
                        masked_labels.append(lbl)
        
        # masked_data.extend(original_data)  
        maskD = maskedDataset((original_data + masked_data), (original_label + masked_labels))
        
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