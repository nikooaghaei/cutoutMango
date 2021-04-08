import torch
import torchvision
import torchvision.transforms as transforms
from util.cutout import Cutout
from util.cutoutFixed16 import CutoutF
from util.mango_try import MANGO_TRY, Cutout_TRY

def set_data(args):
	#### IMAGE PROCESSING ####
	train_transform = transforms.Compose([])

	if args.data_augmentation:
		train_transform.transforms.append(transforms.RandomCrop(32, padding=4))
		train_transform.transforms.append(transforms.RandomHorizontalFlip())

	train_transform.transforms.append(transforms.ToTensor())
	# Bedir's normalization
	bedir_normalize = transforms.Normalize(mean = [ 0.485, 0.456, 0.406 ],
						std  = [ 0.229, 0.224, 0.225 ])
	#UA normalize
	_CIFAR_MEAN, _CIFAR_STD = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
	UA_normalize = transforms.Normalize(_CIFAR_MEAN, _CIFAR_STD)
	#Cutout normalization
	cutout_normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                        std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
	train_transform.transforms.append(cutout_normalize)
	if args.cutout:
		# train_transform.transforms.append(Cutout(n_holes=args.cutout_n_holes, length=args.cutout_len))
		train_transform.transforms.append(Cutout_TRY(n_holes=args.cutout_n_holes, length=args.cutout_len))
	if args.fixedcutout:
		train_transform.transforms.append(Cutout(n_holes=args.cutout_n_holes, length=args.cutout_len))

	test_transform = transforms.Compose([
        transforms.ToTensor(), cutout_normalize])

	#### CREATING TEST/TRAIN DATA
	if args.first_dataset == 'cifar10':
		num_classes = 10
		trainset = torchvision.datasets.CIFAR10(root='../data', train=True,
											download=True, transform=train_transform)
		testset = torchvision.datasets.CIFAR10(root='../data', train=False,
											download=True, transform=test_transform)
	elif args.first_dataset == 'cifar100':
		num_classes = 100
		trainset = torchvision.datasets.CIFAR10(root='../data', train=True,
												download=True, transform=train_transform)
		testset = torchvision.datasets.CIFAR10(root='../data', train=False,
											download=True, transform=test_transform)

	#### CREATING TEST/TRAIN DATA LOADER
	trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
											shuffle=True, num_workers=args.n_workers)
	testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size,
											shuffle=False, num_workers=args.n_workers)

	return trainloader, testloader, num_classes


def set_data_mango(args, model_here):
    	#### IMAGE PROCESSING ####
	train_transform = transforms.Compose([])

	if args.data_augmentation:
		train_transform.transforms.append(transforms.RandomCrop(32, padding=4))
		train_transform.transforms.append(transforms.RandomHorizontalFlip())

	train_transform.transforms.append(transforms.ToTensor())

	#Cutout normalization
	cutout_normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                        std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
	train_transform.transforms.append(cutout_normalize)

	if args.mango:
		train_transform.transforms.append(MANGO_TRY(n_holes=args.cutout_n_holes, length=args.cutout_len,
													model=model_here, device=args.device))

	test_transform = transforms.Compose([
        transforms.ToTensor(), cutout_normalize])

	#### CREATING TEST/TRAIN DATA
	if args.first_dataset == 'cifar10':
		num_classes = 10
		trainset = torchvision.datasets.CIFAR10(root='../data', train=True,
											download=True, transform=train_transform)
		testset = torchvision.datasets.CIFAR10(root='../data', train=False,
											download=True, transform=test_transform)
	elif args.first_dataset == 'cifar100':
		num_classes = 100
		trainset = torchvision.datasets.CIFAR10(root='../data', train=True,
												download=True, transform=train_transform)
		testset = torchvision.datasets.CIFAR10(root='../data', train=False,
											download=True, transform=test_transform)

	#### CREATING TEST/TRAIN DATA LOADER
	trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
											shuffle=True, num_workers=args.n_workers)
	testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size,
											shuffle=False, num_workers=args.n_workers)

	return trainloader, testloader, num_classes