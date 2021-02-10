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
    #        for image in images:
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

########################################new
def train_loop(training_loader, testing_loader, sec_run):
    FLAG_THRESHOLD = False
    threshold = 0.0001
    diff_counter = 0
    diff_limit = 4
    for epoch in range(args.epochs):

       	xentropy_loss_avg = 0.
        correct = 0.
        total = 0.

        prev_acc = 0

        progress_bar = tqdm(training_loader)
        for i, (images, labels) in enumerate(progress_bar):   
            progress_bar.set_description('Epoch ' + str(epoch))

            images = images.cuda()
            labels = labels.cuda()

            #if sec_run and flag:
            cnn.zero_grad()
             #   flag = False
            #else:
            #    cnn.zero_grad()
            pred = cnn(images)

            #########temp

            xentropy_loss = criterion(pred, labels)
            xentropy_loss.backward()
            cnn_optimizer.step()

            xentropy_loss_avg += xentropy_loss.item()

            # Calculate running average of accuracy
            pred = torch.max(pred.data, 1)[1]
            total += labels.size(0)
            correct += (pred == labels.data).sum().item()
            accuracy = correct / total
             
	    # THRESHOLD CHECK #
	    ###################

            if accuracy - prev_acc <= threshold:
                diff_counter += 1
                if diff_counter == diff_limit:
                    FLAG_THRESHOLD = True        
            else:
                diff_counter = 0   
            prev_acc = accuracy

	    ###################
            progress_bar.set_postfix(
                xentropy='%.3f' % (xentropy_loss_avg / (i + 1)),
                acc='%.3f' % accuracy)

    #    main_parts = find_main_part(test_loader)    ##Newwwwwwwww

        test_acc = test(testing_loader)  

        tqdm.write('test_acc: %.3f' % (test_acc))  

        scheduler.step(epoch)  # Use this line for PyTorch <1.4
        # scheduler.step()     # Use this line for PyTorch >=1.4

        row = {'epoch': str(epoch), 'train_acc': str(accuracy), 'test_acc': str(test_acc)} 
        
        if FLAG_THRESHOLD:
             break

    return

if __name__ == '__main__':
    # mp.set_start_method('spawn')
    
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

    args = parser.parse_args()
    

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    cudnn.benchmark = True  # Should make training should go faster for large models

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    test_id = args.dataset + '_' + args.model

    # Image Preprocessing
    if args.dataset == 'svhn':
        normalize = transforms.Normalize(mean=[x / 255.0 for x in[109.9, 109.7, 113.8]],
                                        std=[x / 255.0 for x in [50.1, 50.6, 50.8]])
    else:
        normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                        std=[x / 255.0 for x in [63.0, 62.1, 66.7]])


    ################testtttttttt
    _CIFAR_MEAN, _CIFAR_STD = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
    UA_normalize = transforms.Normalize(_CIFAR_MEAN, _CIFAR_STD)

    #creating train/test transform
    train_transform = transforms.Compose([])

    if args.data_augmentation:
        train_transform.transforms.append(transforms.RandomCrop(32, padding=4))
        train_transform.transforms.append(transforms.RandomHorizontalFlip())

    train_transform.transforms.append(transforms.ToTensor())
    train_transform.transforms.append(normalize)

    if args.cutout:
        train_transform.transforms.append(Cutout(n_holes=args.n_holes, length=args.length))

    test_transform = transforms.Compose([
        transforms.ToTensor(), normalize])

    #creating the dataset for normal train and test
    train_dataset, test_dataset, num_classes = creat_dataset(train_transform, test_transform, 'data/')

    #creating data loaders
    # Data Loader (Input Pipeline)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                            batch_size=args.batch_size,
                                            shuffle=True,
                                            pin_memory=True,
                                            num_workers=2)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                            batch_size=args.batch_size,
                                            shuffle=False,
                                            pin_memory=True,
                                            num_workers=2)

    #creating the model
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

    filename = 'logs/' + test_id + '.csv'
    csv_logger = CSVLogger(args=args, fieldnames=['epoch', 'train_acc', 'test_acc'], filename=filename)

    ###simple testing and training loop
    train_loop(train_loader, test_loader, False)

    torch.save(cnn.state_dict(), 'checkpoints/' + test_id + '.pt')
    csv_logger.close()

    del train_loader
    #################################################newwwwwwwwwwwwwwwwwww
    if args.mango:
        retrain_transform = transforms.Compose([])

        if args.data_augmentation:
            retrain_transform.transforms.append(transforms.RandomCrop(32, padding=4))
            retrain_transform.transforms.append(transforms.RandomHorizontalFlip())

        if args.cutout:
            retrain_transform.transforms.append(Cutout(n_holes=args.n_holes, length=args.length))

        retrain_transform.transforms.append(transforms.ToTensor())
        retrain_transform.transforms.append(normalize)
        retrain_transform.transforms.append(Mango(cnn))

        retrain_dataset, test_dataset, num_classes = creat_dataset(retrain_transform, test_transform, 'data/MANGO')

        retrain_loader = torch.utils.data.DataLoader(dataset=retrain_dataset,
                                                batch_size=args.batch_size,###############????same?
                                                shuffle=True,
    #                                            pin_memory=True,
                                                num_workers=2)

        MNG_filename = 'logs/MANGO/' + test_id + '.csv'
        MNG_csv_logger = CSVLogger(args=args, fieldnames=['epoch', 'train_acc', 'test_acc'], filename=MNG_filename)

        train_loop(retrain_loader,test_loader, True)

        torch.save(cnn.state_dict(), 'checkpoints/MANGO/' + test_id + '.pt')
        MNG_csv_logger.close()