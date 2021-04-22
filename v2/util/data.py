import torch
import torchvision
import torchvision.transforms as transforms
from util.cutout import Cutout
from util.mango_try import MANGO_FIXED, Cutout_FIXED,MANGO_CUT, MANGOCut_MaskDiff, MANGO_CUT_S, OrigMANGO_CUT

def set_data(args, trained_model = None, is_mango = False):
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
		train_transform.transforms.append(Cutout(n_holes=args.cutout_n_holes, length=args.cutout_len))
	if is_mango:
		train_transform.transforms.append(MANGOCut_MaskDiff(model=trained_model, args=args))

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
	if is_mango:
		n_workers = 2
	else:
		n_workers = 0
	#### CREATING TEST/TRAIN DATA LOADER
	trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
											shuffle=True, num_workers=n_workers)
	testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size,
											shuffle=False, num_workers=n_workers)

	return trainloader, testloader, num_classes