# run train.py --dataset cifar10 --model resnet18 --data_augmentation --cutout --length 16
# run train.py --dataset cifar100 --model resnet18 --data_augmentation --cutout --length 8
# run train.py --dataset svhn --model wideresnet --learning_rate 0.01 --epochs 160 --cutout --length 20
import sys
import pdb
import argparse
import numpy as np
from tqdm import tqdm
import random
import copy

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import MultiStepLR

from torchvision.utils import make_grid
from torchvision.utils import save_image
from torchvision import datasets, transforms

from util.misc import CSVLogger
from util.cutout import Cutout
from util.fixed_mng import Fixed_MNG
from util.mango import Mango

from model.resnet import ResNet18
from model.wide_resnet import WideResNet

from multiprocessing import set_start_method
import multiprocessing as mp

#########################new
def find_main_part(loader):######newwwwwwwwww
    cnn.eval()    ########????????????
    main_nodes = []
    for images, labels in loader:
        images = images.cuda()
        labels = labels.cuda()

        with torch.no_grad():
            # for image in images:
            for i in range (128):
                #i = random.randint(0, 128)
                mng = Mango(cnn, images[i])
                main_nodes.append(mng())

                ###printing results
                mng.show_chain(str(i))
                #########test part for one img only:
            cnn.train() #####???
            return main_nodes


    cnn.train() #####???
    return main_nodes

def test(loader):
    cnn.eval()    # Change model to 'eval' mode (BN uses moving mean/var).
    correct = 0.
    total = 0.
    for images, labels in loader:
        images = images.cuda()
        labels = labels.cuda()
        
        with torch.no_grad():
            pred = cnn(images)

        pred = torch.max(pred.data, 1)[1]
        total += labels.size(0)
        correct += (pred == labels).sum().item()

    val_acc = correct / total
    cnn.train()
    return val_acc

########################################new
def creat_dataset(training_transform, testing_transform, root):
    if args.dataset == 'cifar10':
        num_classes = 10
        training_dataset = datasets.CIFAR10(root=root,
                                        train=True,
                                        transform=training_transform,
                                        download=True)

        testing_dataset = datasets.CIFAR10(root=root,
                                        train=False,
                                        transform=testing_transform,
                                        download=True)
    elif args.dataset == 'cifar100':
        num_classes = 100
        training_dataset = datasets.CIFAR100(root=root,
                                        train=True,
                                        transform=training_transform,
                                        download=True)

        testing_dataset = datasets.CIFAR100(root=root,
                                        train=False,
                                        transform=testing_transform,
                                        download=True)
    elif args.dataset == 'svhn':
        num_classes = 10
        training_dataset = datasets.SVHN(root=root,
                                    split='train',
                                    transform=training_transform,
                                    download=True)
        extra_dataset = datasets.SVHN(root=root,
                                    split='extra',
                                    transform=training_transform,
                                    download=True)
        
        # Combine both training splits (https://arxiv.org/pdf/1605.07146.pdf)
        data = np.concatenate([training_dataset.data, extra_dataset.data], axis=0)
        labels = np.concatenate([training_dataset.labels, extra_dataset.labels], axis=0)
        training_dataset.data = data
        training_dataset.labels = labels

        testing_dataset = datasets.SVHN(root=root,
                                    split='test',
                                    transform=testing_transform,
                                    download=True)


    return training_dataset, testing_dataset, num_classes

#################################fixed
def train_fixed(training_loader, testing_loader):
    for epoch in range(args.epochs):

       	xentropy_loss_avg = 0.
        correct = 0.
        total = 0.
        progress_bar = tqdm(training_loader)
        for i, (images, labels) in enumerate(progress_bar):   
            progress_bar.set_description('Epoch ' + str(epoch))
            print("start")
            images = images.cuda()
            labels = labels.cuda()

            # cnn.eval()
  
            # mng_imgs=[]
            # img_num = 0
            # for index in range(len(images)):
            #     # mng = Fixed_MNG(length = args.length, model = cnn)
            #     mng = Cutout(length = args.length, model = cnn)
            #     res = mng(images[index])
            #     if (epoch == 0 or epoch == 199) and i == 0:
            #         save_image(res, 'data/fixed_mng/' + test_id + '/epoch' + str(epoch) + '/batch' + str(i) + '/' + str(img_num) + '_label' + str(labels[index].item()) + '.png')
            #     img_num = img_num +1
            #     mng_imgs.append(res)

            # mng_imgs = torch.stack(mng_imgs)
            # print("end")
            # cnn.train()

            cnn.zero_grad()
            # pred = cnn(mng_imgs)
            # print("after pred")
            # else:
            pred = cnn(images)

            xentropy_loss = criterion(pred, labels)
            xentropy_loss.backward()
            cnn_optimizer.step()

            xentropy_loss_avg += xentropy_loss.item()

            # Calculate running average of accuracy
            pred = torch.max(pred.data, 1)[1]
            total += labels.size(0)
            correct += (pred == labels.data).sum().item()
            accuracy = correct / total
        
            progress_bar.set_postfix(
                xentropy='%.3f' % (xentropy_loss_avg / (i + 1)),
                acc='%.3f' % accuracy)

        # main_parts = find_main_part(test_loader)    ##Newwwwwwwww
        test_acc = test(testing_loader)  
        tqdm.write('test_acc: %.3f' % (test_acc))  

        # scheduler.step(epoch)  # Use this line for PyTorch <1.4
        scheduler.step()     # Use this line for PyTorch >=1.4

        row = {'epoch': str(epoch), 'train_acc': str(accuracy), 'test_acc': str(test_acc)} 
        csv_logger.writerow(row)
    return

########################################new
def train_loop(training_loader, testing_loader):
    for epoch in range(args.epochs):

       	xentropy_loss_avg = 0.
        correct = 0.
        total = 0.

        progress_bar = tqdm(training_loader)
        for i, (images, labels) in enumerate(progress_bar):   
            progress_bar.set_description('Epoch ' + str(epoch))

            images = images.cuda()
            labels = labels.cuda()

            # print("start")

            # if sec_run:

            # print("Creating MANGO data..")
            cnn.eval()
  
            mng_imgs=[]
            img_num = 0
            for index in range(len(images)):
                mng = Mango(cnn)
                res = mng(images[index])
            #         if mng.res.mask_loc:	#if mask was not None (image has changed)
                if (epoch == 0 or epoch == 199) and i == 390:
                    save_image(res, 'data/MANGO/' + test_id + '/epoch' + str(epoch) + '/batch' + str(i) + '/' + str(img_num) + '_label' + str(labels[index].item()) + '.png')
                img_num = img_num +1
                mng_imgs.append(res)

            mng_imgs = torch.stack(mng_imgs)
            
            cnn.train()

            cnn.zero_grad()
            pred = cnn(mng_imgs)
            # else:
            # pred = cnn(images)

            xentropy_loss = criterion(pred, labels)
            xentropy_loss.backward()
            cnn_optimizer.step()

            xentropy_loss_avg += xentropy_loss.item()

            # Calculate running average of accuracy
            pred = torch.max(pred.data, 1)[1]
            total += labels.size(0)
            correct += (pred == labels.data).sum().item()
            accuracy = correct / total
        
            progress_bar.set_postfix(
                xentropy='%.3f' % (xentropy_loss_avg / (i + 1)),
                acc='%.3f' % accuracy)

        # main_parts = find_main_part(test_loader)    ##Newwwwwwwww
        # print("before test")
        test_acc = test(testing_loader)  
        # print("after test")
        tqdm.write('test_acc: %.3f' % (test_acc))  

        # scheduler.step(epoch)  # Use this line for PyTorch <1.4
        scheduler.step()     # Use this line for PyTorch >=1.4

        row = {'epoch': str(epoch), 'train_acc': str(accuracy), 'test_acc': str(test_acc)} 
        csv_logger.writerow(row)
    return

########################################new
def train_loop2(training_loader, testing_loader):

    ##############creating new data
    masked_imgs=[]
    all_labels= []
   
    temp = 3
    
    cnn.eval()

    print("creating MANGO data ...")  
    
    img_num = 0	#used for naming new datapoints    

    progress_bar = tqdm(training_loader)
    for i, (images, labels) in enumerate(progress_bar):
  
        images = images.cuda()
        labels = labels.cuda()
       
        for index in range(len(images)):
            with torch.no_grad():
                mng = Mango(cnn)
                res = mng(images[index])
            if mng.res.mask_loc:       #if mask was not None (image has changed)
                masked_imgs.append(res)
                all_labels.append(labels[index])
                save_image(res, 'data/MANGO/' + test_id + '/' + str(img_num) + '_label' + str(labels[index].item()) + '.png')
                img_num = img_num + 1
        # temp = temp - 1
        # if temp == 0:
        #    break
    # all_labels = torch.stack(all_labels)
    # masked_imgs = torch.stack(masked_imgs)

    print("starting retraining ...")

    cnn.train()

    ############starting epochs
    for epoch in range(args.epochs):
        progress_bar.set_description('Epoch ' + str(epoch))

        xentropy_loss_avg = 0.
        correct = 0.
        total = 0.

        progress_bar = tqdm(range(0, len(masked_imgs), 128))
        for k in (progress_bar):
            batch_imgs= []
            batch_labels = []

            for j in range(k, k+128):
                if j >= len(masked_imgs):
                    break
                batch_imgs.append(masked_imgs[j])
                batch_labels.append(all_labels[j])

            batch_imgs = torch.stack(batch_imgs)
            batch_labels = torch.stack(batch_labels)

            cnn.zero_grad()
            pred = cnn(batch_imgs)

            xentropy_loss = criterion(pred, batch_labels)
            xentropy_loss.backward()
            cnn_optimizer.step()

            xentropy_loss_avg += xentropy_loss.item()

            # Calculate running average of accuracy
            pred = torch.max(pred.data, 1)[1]
            total += batch_labels.size(0)
            correct += (pred == batch_labels.data).sum().item()
            accuracy = correct / total

            progress_bar.set_postfix(
                xentropy='%.3f' % (xentropy_loss_avg / (i + 1)),
                acc='%.3f' % accuracy)

        # main_parts = find_main_part(test_loader)    ##Newwwwwwwww
 
        test_acc = test(testing_loader)
 
        tqdm.write('test_acc: %.3f' % (test_acc))

        # scheduler.step(epoch)     # Use this line for PyTorch <1.4
        scheduler.step()            # Use this line for PyTorch >=1.4

        row = {'epoch': str(epoch), 'train_acc': str(accuracy), 'test_acc': str(test_acc)}
        MNG_csv_logger.writerow(row)

    return



def main():
    # global model_options
    # global dataset_options
    # global parser 
    global test_id
    global scheduler
    global csv_logger
    global MNG_csv_logger
    global args
    global cnn
    global criterion
    global cnn_optimizer
    model_options = ['resnet18', 'wideresnet']
    dataset_options = ['cifar10', 'cifar100', 'svhn']

    parser = argparse.ArgumentParser(description='CNN')
    parser.add_argument('--dataset', '-d', default='cifar10',
                        choices=dataset_options)
    parser.add_argument('--model', '-a', default='resnet18',
                        choices=model_options)
    parser.add_argument('--batch_size', type=int, default=128,
                        help='input batch size for training (default: 128)')
    parser.add_argument('--epochs', type=int, default=1,
                        help='number of epochs to train (default: 200)')
    parser.add_argument('--learning_rate', type=float, default=0.1,
                        help='learning rate')
    parser.add_argument('--data_augmentation', action='store_true', default=False,
                        help='augment data by flipping and cropping')
    parser.add_argument('--cutout', action='store_true', default=False,
                        help='apply cutout')
    parser.add_argument('--n_holes', type=int, default=1,
                        help='number of holes to cut out from image')
    parser.add_argument('--length', type=int, default=16,
                        help='length of the holes')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed (default: 1)')

    ######################new
    parser.add_argument('--mango', action='store_true', default=False,  ##newwwww
                        help='apply mango')
    parser.add_argument('--experiment_type', type=str, default='default',
                        help='default: default values with no retraining')
    parser.add_argument('--n_workers', type=int, default=2,
			help='number of workers')


    args = parser.parse_args()
    
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    cudnn.benchmark = True  # Should make training should go faster for large models

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    test_id = args.dataset + '_' + args.model

    print(args)

    # Image Preprocessing
    if args.dataset == 'svhn':
        normalize = transforms.Normalize(mean=[x / 255.0 for x in[109.9, 109.7, 113.8]],
                                        std=[x / 255.0 for x in [50.1, 50.6, 50.8]])
    else:
        normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                        std=[x / 255.0 for x in [63.0, 62.1, 66.7]])


    ################ testtttttttt
    _CIFAR_MEAN, _CIFAR_STD = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
    UA_normalize = transforms.Normalize(_CIFAR_MEAN, _CIFAR_STD)

    # creating train/test transform
    train_transform = transforms.Compose([])

    if args.data_augmentation:
        train_transform.transforms.append(transforms.RandomCrop(32, padding=4))
        train_transform.transforms.append(transforms.RandomHorizontalFlip())

    train_transform.transforms.append(transforms.ToTensor())
    train_transform.transforms.append(normalize)

    if args.cutout:
        train_transform.transforms.append(Cutout(n_holes=args.n_holes, length=args.length))
        # train_transform.transforms.append(Fixed_MNG(length=args.length, model = cnn))

    test_transform = transforms.Compose([
        transforms.ToTensor(), normalize])

    # creating the dataset for normal train and test
    train_dataset, test_dataset, num_classes = creat_dataset(train_transform, test_transform, 'data/fixed_mng/')
    
    # creating data loaders
    # Data Loader (Input Pipeline)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                            batch_size=args.batch_size,
                                            shuffle=True,
                                            pin_memory=True,
                                            num_workers=args.n_workers)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                            batch_size=args.batch_size,
                                            shuffle=False,
                                            pin_memory=True,
                                            num_workers=args.n_workers)

    # creating the model
    if args.model == 'resnet18':
        cnn = ResNet18(num_classes=num_classes)
    elif args.model == 'wideresnet':
        if args.dataset == 'svhn':
            cnn = WideResNet(depth=16, num_classes=num_classes, widen_factor=8,
                            dropRate=0.4)
        else:
            cnn = WideResNet(depth=28, num_classes=num_classes, widen_factor=10,
                            dropRate=0.3)

    cnn = cnn.cuda()
    criterion = nn.CrossEntropyLoss().cuda()
    cnn_optimizer = torch.optim.SGD(cnn.parameters(), lr=args.learning_rate,
                                    momentum=0.9, nesterov=True, weight_decay=5e-4)

    if args.dataset == 'svhn':
        scheduler = MultiStepLR(cnn_optimizer, milestones=[80, 120], gamma=0.1)
    else:
        scheduler = MultiStepLR(cnn_optimizer, milestones=[60, 120, 160], gamma=0.2)

    filename = 'logs/fixed_mng' + test_id + '.csv'
    csv_logger = CSVLogger(args=args, fieldnames=['epoch', 'train_acc', 'test_acc'], filename=filename)


    ### simple testing and training loop
    train_fixed(train_loader, test_loader)

    torch.save(cnn.state_dict(), 'checkpoints/fixed_mng' + test_id + '.pt')
    csv_logger.close()

    #################################################newwwwwwwwwwwwwwwwwww
    if args.mango:
        # retrain_transform = transforms.Compose([])

        # if args.data_augmentation:
        #     retrain_transform.transforms.append(transforms.RandomCrop(32, padding=4))
        #     retrain_transform.transforms.append(transforms.RandomHorizontalFlip())

        # if args.cutout:
        #     retrain_transform.transforms.append(Cutout(n_holes=args.n_holes, length=args.length))

        # retrain_transform.transforms.append(transforms.ToTensor())
        # retrain_transform.transforms.append(normalize)
        # retrain_transform.transforms.append(Mango(cnn))

        # retrain_dataset, test_dataset, num_classes = creat_dataset(retrain_transform, test_transform, 'data/MANGO')

        # retrain_loader = torch.utils.data.DataLoader(dataset=retrain_dataset,
        #                                     batch_size=args.batch_size,###############????same?
        #                                     shuffle=True,
        #                                     pin_memory=True,
        #                                     num_workers=args.n_workers)

        MNG_filename = 'logs/MANGO/' + test_id + '.csv'
        MNG_csv_logger = CSVLogger(args=args, fieldnames=['epoch', 'train_acc', 'test_acc'], filename=MNG_filename)

        # train_loop(retrain_loader,test_loader, True)
        train_loop2(train_loader, test_loader)

        torch.save(cnn.state_dict(), 'checkpoints/MANGO/' + test_id + '.pt')
        MNG_csv_logger.close()

if __name__ == '__main__':
    # mp.set_start_method('spawn')
    main()
