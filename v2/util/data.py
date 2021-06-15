import torch
import torchvision
import torchvision.transforms as transforms
from util.cutout import Cutout
from util.mango_try import ForcedMANGO, ForcedMANGO_gt, OrigMANGO, MngCut_RandomColor, MANGO_CUT,\
	Mng_RandColor, ForcedMngCut, ForcedMngCut_gt, MngCut_gt, OrigMANGO_gt

def set_data(args, trained_model = None, is_mango = False):
	if args.first_dataset == 'cifar10':
		num_classes = 10
	elif args.first_dataset == 'cifar100':
		num_classes = 100
	if not is_mango and args.first_model_load_path:	####using pretrained model so no need for making dataset for phase1
		return None, None, num_classes
		
	#### IMAGE PROCESSING ####
	train_transform = transforms.Compose([])

	if args.data_augmentation:
		train_transform.transforms.append(transforms.RandomCrop(32, padding=4))
		train_transform.transforms.append(transforms.RandomHorizontalFlip())

	train_transform.transforms.append(transforms.ToTensor())
	
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
		if args.origmng:
			if args.forced and args.gt:
				train_transform.transforms.append(ForcedMANGO_gt(model=trained_model, args=args))
			elif args.forced:
				train_transform.transforms.append(ForcedMANGO(model=trained_model, args=args))
			elif args.gt:
				train_transform.transforms.append(OrigMANGO_gt(model=trained_model, args=args))		
			else:
				train_transform.transforms.append(OrigMANGO(model=trained_model, args=args))
		else:
			if args.forced and args.gt:
				train_transform.transforms.append(ForcedMngCut_gt(model=trained_model, args=args))
			elif args.forced:
				train_transform.transforms.append(ForcedMngCut(model=trained_model, args=args))
			elif args.gt:
				train_transform.transforms.append(MngCut_gt(model=trained_model, args=args))		
			else:
				train_transform.transforms.append(MANGO_CUT(model=trained_model, args=args))

	test_transform = transforms.Compose([
        transforms.ToTensor(), cutout_normalize])

	#### CREATING TEST/TRAIN DATA
	if args.mng_dataset == 'cifar10':
		trainset = torchvision.datasets.CIFAR10(root='../data', train=True,
											download=True, transform=train_transform)
		testset = torchvision.datasets.CIFAR10(root='../data', train=False,
											download=True, transform=test_transform)
	elif args.mng_dataset == 'cifar100':
		trainset = torchvision.datasets.CIFAR100(root='../data', train=True,
												download=True, transform=train_transform)
		testset = torchvision.datasets.CIFAR100(root='../data', train=False,
											download=True, transform=test_transform)
	# if is_mango:
	# n_workers = 2
	# else:
	# 	n_workers = 0
	#### CREATING TEST/TRAIN DATA LOADER
	trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
											shuffle=True, num_workers=args.n_workers)
	testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size,
											shuffle=False, num_workers=args.n_workers)

	return trainloader, testloader, num_classes