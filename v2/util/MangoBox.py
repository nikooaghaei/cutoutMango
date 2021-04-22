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
from torchvision.utils import save_image

def load_from(path):
    with open(path, "rb") as fp:
        return pickle.load(fp)
def run_mango(model, trainloader, args):
    if args.mng_load_data_path:
        new_train = load_from(args.mng_load_data_path)
        if not isinstance(new_train, maskedDataset):
            raise ValueError('Inappropriate type loaded: {} for the data \
                whereas a maskedDataset is expected'.format(type(new_train)))
    else:
        mango = Mango(model, trainloader, args)
        new_train = mango.create_dataset()
        # Determine number of classes in mango dataset
    if args.mng_dataset == 'cifar10':
        num_classes = 10
    elif args.mng_dataset == 'cifar100':
        num_classes = 100
    
    return torch.utils.data.DataLoader(new_train, batch_size=args.batch_size,
                                       shuffle=True, 
                                       num_workers=args.n_workers), num_classes
class Mango(object):
    """Mask out most important part of an image.
    Args:
        model (torch.model): Pytorch model to mask according to
        data (list - [(img, lbl), ...]): Image data to mask
        folder_name (str): Name of the folder to save the images on
        n_branches (int): Each node is divided to n_branches nodes -ongoing
    """
    def __init__(self, model, data, args):
        self.model = model
        self.n_branches = args.mng_n_branches
        self.init_length = args.mng_init_len
        self.data = data
        
        self.path = "data/MANGO/" + args.experiment_name
        self.device = args.device
    ################################test cutout here:
    def cutout2(self, img):
        """
        original cutout being applied but not as a data transform!
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(1):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.init_length // 2, 0, h)
            y2 = np.clip(y + self.init_length // 2, 0, h)
            x1 = np.clip(x - self.init_length // 2, 0, w)
            x2 = np.clip(x + self.init_length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img

    def _run_masking(self, img):
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
        # change to decide how deep to go. E.g. to go down to 1 pixel mask size set it to 1
        min_mask_len = 0  #Or w//4 as our images are squar
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
        ###############new
        # value = nnf.softmax(pred, dim = 1)[0][label]
        ##################
        min_prob = value
        label = index[0]
        res = img
        while mask_len > min_mask_len:
            masks = np.ones((self.n_branches, h, w), np.float32)
            
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
                    temp_img = masked_img.view(1,3,32,32).to(self.device)
                    try:
                        pred = self.model(temp_img).to(self.device)
                    except:
                        pred = self.model(temp_img)
                softmax_prob = nnf.softmax(pred, dim = 1)
                prob = softmax_prob[0][label]

                if prob < min_prob: ######treshold comes here
                    min_prob = prob ###??pointer
                    expanding_node = m
                    res = masked_img
            if expanding_node == -1:
                # save_image(res, str(count) + '_label_' + str(label) + '.png')
                return res, (mask_len * 2) % h
            mask_len = mask_len//2
            if expanding_node == 0:
                y3 = y2
                y2 = np.clip(y1 + mask_len, 0, y2)
                x3 = x2
                x2 = np.clip(x1+mask_len, 0, x2)
            elif expanding_node == 1:
                y3 = y2
                y2 = np.clip(y1 + mask_len, 0, y2)
                x1 = x2
                x2 = np.clip(x2+mask_len, 0, x3)
            elif expanding_node == 2:
                y1 = y2
                y2 = np.clip(y2 + mask_len, 0, y3)
                x3 = x2
                x2 = np.clip(x1+mask_len, 0, x2)
            elif expanding_node == 3:
                y1 = y2
                y2 = np.clip(y2 + mask_len, 0, y3)
                x1 = x2
                x2 = np.clip(x2+mask_len, 0, x3)

        return res, (mask_len * 2) % h

    def _run_masking3(self, img):
        """
        3x3 MANGO

        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with one hole of dimension 
            ((mask_len * 2) % h x (mask_len * 2) % h) cut out of it.
            Integer (mask_len * 2) % h : Length of the mask cut out of the image.
        """
        h = img.size(1)
        w = img.size(2)
        # change to decide how deep to go. E.g. to go down to 1 pixel mask size set it to 1
        min_mask_len = 0  #Or w//4 as our images are squar
        mask_len = self.init_length
        y1 = np.clip(0, 0, h)
        y2 = np.clip(mask_len, 0, h)
        y3 = np.clip(2 * mask_len, 0, h)
        y4 = np.clip(h, 0, h)
        x1 = np.clip(0, 0, w)
        x2 = np.clip(mask_len, 0, w)
        x3 = np.clip(2*mask_len, 0, w)
        x4 = np.clip(w, 0, w)
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
        res = img
        while mask_len > min_mask_len:
            masks = np.ones((self.n_branches, h, w), np.float32)
            
            masks[0][y1: y2, x1: x2] = 0. # or any other colors for mask
            masks[1][y1: y2, x2: x3] = 0.
            masks[2][y1: y2, x3: x4] = 0.
            masks[3][y2: y3, x1: x2] = 0.
            masks[4][y2: y3, x2: x3] = 0.
            masks[5][y2: y3, x3: x4] = 0.
            masks[6][y3: y4, x1: x2] = 0.
            masks[7][y3: y4, x2: x3] = 0.
            masks[8][y3: y4, x3: x4] = 0.
            masks = torch.from_numpy(masks)
            expanding_node = -1 # referring to original img
            
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
                    expanding_node = m
                    res = masked_img

            if expanding_node == -1:
                # save_image(res, str(count)+ '.png')
                return res, (mask_len * 3) % h

            mask_len = mask_len//3
            if expanding_node == 0:
                y3 = np.clip(y1 + 2 * mask_len, 0, y2)
                y4 = y2
                y2 = np.clip(y1 + mask_len, 0, y2)
                x3 = np.clip(x1 + 2*mask_len, 0, x2)
                x4 = x2
                x2 = np.clip(x1+mask_len, 0, x2)
            elif expanding_node == 1:
                y3 = np.clip(y1 + 2 * mask_len, 0, y2)
                y4 = y2
                y2 = np.clip(y1 + mask_len, 0, y2)
                x1 = x2
                x4 = x3
                x2 = np.clip(x2+mask_len, 0, x3)
                x3 = np.clip(x1+2*mask_len, 0, x3)
            elif expanding_node == 2:
                y3 = np.clip(y1 + 2 * mask_len, 0, y2)
                y4 = y2
                y2 = np.clip(y1 + mask_len, 0, y2)
                x1 = x3
                x2 = np.clip(x3+mask_len, 0, x4)
                x3 = np.clip(x3+2 *mask_len, 0, x4)
            elif expanding_node == 3:
                y1 = y2
                y4 = y3
                y2 = np.clip(y2 + mask_len, 0, y3)
                y3 = np.clip(y1 + 2*mask_len, 0, y3)
                x3 = np.clip(x1 + 2*mask_len, 0, x2)
                x4 = x2
                x2 = np.clip(x1+mask_len, 0, x2)
            elif expanding_node == 4:
                y1 = y2
                y4 = y3
                y2 = np.clip(y2 + mask_len, 0, y3)
                y3 = np.clip(y1 + 2*mask_len, 0, y3)
                x1 = x2
                x4 = x3
                x2 = np.clip(x2+mask_len, 0, x3)
                x3 = np.clip(x1+2*mask_len, 0, x3)
            elif expanding_node == 5:
                y1 = y2
                y4 = y3
                y2 = np.clip(y2 + mask_len, 0, y3)
                y3 = np.clip(y1 + 2*mask_len, 0, y3)
                x1 = x3
                x2 = np.clip(x3+mask_len, 0, x4)
                x3 = np.clip(x3+2 *mask_len, 0, x4)
            elif expanding_node == 6:
                y1 = y3
                y2 = np.clip(y3 + mask_len, 0, y4)
                y3 = np.clip(y3 + 2*mask_len, 0, y4)
                x3 = np.clip(x1 + 2*mask_len, 0, x2)
                x4 = x2
                x2 = np.clip(x1+mask_len, 0, x2)
            elif expanding_node == 7:
                y1 = y3
                y2 = np.clip(y3 + mask_len, 0, y4)
                y3 = np.clip(y3 + 2*mask_len, 0, y4)
                x1 = x2
                x4 = x3
                x2 = np.clip(x2+mask_len, 0, x3)
                x3 = np.clip(x1+2*mask_len, 0, x3)
            elif expanding_node == 8:
                y1 = y3
                y2 = np.clip(y3 + mask_len, 0, y4)
                y3 = np.clip(y3 + 2*mask_len, 0, y4)
                x1 = x2
                x4 = x3
                x2 = np.clip(x2+mask_len, 0, x3)
                x3 = np.clip(x1+2*mask_len, 0, x3)

        return res, (mask_len * 3) % h

    def _run_masking8(self, img):#######WRONG FUNC#######yi and xi variables should be fixed!!!!!!
        #TODO
        """
        version of MANGo wheremain part is chosen among all four 16x16 and all thirty two 8x8 children masks.
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with one hole of dimension 
            ((mask_len * 2) % h x (mask_len * 2) % h) cut out of it.
            Integer (mask_len * 2) % h : Length of the mask cut out of the image.
        """
        h = img.size(1)
        w = img.size(2)
        # change to decide how deep to go. E.g. to go down to 1 pixel mask size set it to 1
        # min_mask_len = 0  #Or w//4 as our images are squar
        mask_len = self.init_length

        no_mask = True
        first_layer = False
        second_layer = False


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
        ###############new
        # value = nnf.softmax(pred, dim = 1)[0][label]
        ##################
        min_prob = value
        label = index[0]
        res = img
        # while mask_len > min_mask_len:
        masks = np.ones((self.n_branches, h, w), np.float32)

        masks[0][y1: y2, x1: x2] = 0. # or any other colors for mask
        masks[1][y1: y2, x2: x3] = 0.
        masks[2][y2: y3, x1: x2] = 0.
        masks[3][y2: y3, x2: x3] = 0.

        masks = torch.from_numpy(masks)

        # expanding_node = -1 # referring to original img

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

            if prob > min_prob: ######treshold comes here
                no_mask = False
                first_layer = True
                min_prob = prob ###??pointer
                # expanding_node = m
                res = masked_img

        # if expanding_node == -1:
        #     # save_image(res, str(count) + '_label_' + str(label) + '.png')
        #     return res, (mask_len * 2) % h

        mask_len = mask_len//2
        masks2 = np.ones((16, h, w), np.float32)
        # if expanding_node == 0:
        y1 = np.clip(y1, 0, y1)
        y2 = np.clip(y1 + mask_len, 0, y2)
        y3 = np.clip(y2, 0, y2)
        x1 = np.clip(x1, 0, x1)
        x2 = np.clip(x1+mask_len, 0, x2)
        x3 = np.clip(x2, 0, x2)

        masks2[0][y1: y2, x1: x2] = 0. # or any other colors for mask
        masks2[1][y1: y2, x2: x3] = 0.
        masks2[2][y2: y3, x1: x2] = 0.
        masks2[3][y2: y3, x2: x3] = 0.
        # elif expanding_node == 1:
        y1 = np.clip(y1, 0, y1)
        y2 = np.clip(y1 + mask_len, 0, y2)
        y3 = np.clip(y2, 0, y2)
        x1 = np.clip(x2, 0, x2)
        x2 = np.clip(x2+mask_len, 0, x3)
        x3 = np.clip(x3, 0, x3)

        masks2[4][y1: y2, x1: x2] = 0. # or any other colors for mask
        masks2[5][y1: y2, x2: x3] = 0.
        masks2[6][y2: y3, x1: x2] = 0.
        masks2[7][y2: y3, x2: x3] = 0.
        # elif expanding_node == 2:
        y1 = np.clip(y2, 0, y2)
        y2 = np.clip(y2 + mask_len, 0, y3)
        y3 = np.clip(y3, 0, y3)
        x1 = np.clip(x1, 0, x1)
        x2 = np.clip(x1+mask_len, 0, x2)
        x3 = np.clip(x2, 0, x2)

        masks2[8][y1: y2, x1: x2] = 0. # or any other colors for mask
        masks2[9][y1: y2, x2: x3] = 0.
        masks2[10][y2: y3, x1: x2] = 0.
        masks2[11][y2: y3, x2: x3] = 0.
        # elif expanding_node == 3:
        y1 = np.clip(y2, 0, y2)
        y2 = np.clip(y2 + mask_len, 0, y3)
        y3 = np.clip(y3, 0, y3)
        x1 = np.clip(x2, 0, x2)
        x2 = np.clip(x2+mask_len, 0, x3)
        x3 = np.clip(x3, 0, x3)

        masks2[12][y1: y2, x1: x2] = 0. # or any other colors for mask
        masks2[13][y1: y2, x2: x3] = 0.
        masks2[14][y2: y3, x1: x2] = 0.
        masks2[15][y2: y3, x2: x3] = 0.

        masks2 = torch.from_numpy(masks2)

        for mask in masks2:
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

            if prob > min_prob: ######treshold comes here
                no_mask = False
                first_layer = False
                second_layer = True
                min_prob = prob ###??pointer
                res = masked_img

        if no_mask:
            mask_len = 0
        elif first_layer:
            mask_len = 16
        else:
            mask_len = 8
        return res, mask_len

    def create_dataset(self):
        '''
        TODO Add, Args - returns - description
        '''
        print("Beginnnig data processing...")
        original_data = []
        original_label = []
        masked_data = [] 
        masked_labels = []
        cnt = 0
        self.model.eval()
        for (images, labels) in tqdm(self.data):
            # for each batch
            # [(img_1, lbl_1), (img_2, lbl_2), ...]
            for img, lbl in zip(images, labels):
                original_data.append(img)   #TODO
                original_label.append(lbl)
                with torch.no_grad():
                    masked_img = self.cutout2(img)
                    # masked_img, mask = self.mngcut(img)

                    # Path('./data/masks/').mkdir(parents=True, exist_ok=True)
                    # with open('./data/masks/mngcut.txt', 'a') as f:
                    #     print(mask, file=f)

                    # if mask:
                    masked_data.append(masked_img)
                    masked_labels.append(lbl)
                    cnt= cnt+1
        self.model.train()
        # masked_data.extend(original_data)  
        # maskD = maskedDataset((original_data + masked_data), (original_label + masked_labels))
        ###############just masked data:
        maskD = maskedDataset((masked_data), (masked_labels))
        
        # create data/MANGO folder if needed
        print("data/MANGO/ created...")
        Path("data/MANGO/").mkdir(parents=True, exist_ok=True)
        with open(self.path + '.txt', "wb") as fp:
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