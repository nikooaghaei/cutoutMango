import argparse

import torch
import torch.backends.cudnn as cudnn

from util.MangoBox import load_from, run_mango
from util.model_tools import train  # , train_simple
from util.data import set_data

# Increasing worker limit -seems to be necessary in some situations
# check for more: https://github.com/pytorch/pytorch/issues/973#issuecomment-346405667
import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))


model_options = ['basemodel', 'resnet18', 'wideresnet', 'vgg16']
dataset_options = ['cifar10', 'cifar100']

parser = argparse.ArgumentParser(description='CNN')
parser.add_argument('--first_dataset', '-d', default='cifar10',
                    choices=dataset_options)
parser.add_argument('--first_model', '-m', default='resnet18',
                    choices=model_options)
parser.add_argument('--mng_dataset', '-md', default='cifar10',
                    choices=dataset_options)
parser.add_argument('--mng_model', '-mm', default='resnet18',
                    choices=model_options)
parser.add_argument('--first_model_load_path', '-mlp', default='',
                    help='path for both models to load from(default: None)')
parser.add_argument('--model_save_path', '-sp1', default='',
                    help='path to save both models in models/args.model_save_path (default: None)')
parser.add_argument('--mng_load_data_path', '-ld', default='',
                    help='path for MANGO data to load (default: None)')
parser.add_argument('--mng_save_data_path', '-sd', default='',
                    help='path to save mango data in data/MANGO/mng_save_data (default: None)')
parser.add_argument('--batch_size', type=int, default=128,
                    help='input batch size for training (default: 128)')
parser.add_argument('--n_epochs', type=int, default=1,
                    help='number of epochs to train (default: 200)')
parser.add_argument('--learning_rate', '-lr', type=float, default=0.01,
                    help='learning rate')
parser.add_argument('--data_augmentation', action='store_true', default=False,
                    help='augment data by flipping and cropping')
parser.add_argument('--cutout', action='store_true', default=False,
                    help='apply Cutout')
parser.add_argument('--cutout_n_holes', type=int, default=1,
                    help='number of holes to cut out from image in Cutout')
parser.add_argument('--cutout_len', type=int, default=16,
                    help='length of the holes in Cutout')
parser.add_argument('--seed', type=int, default=0,
                    help='random seed (default: 1)')
parser.add_argument('--mango', action='store_true', default=False,
                    help='apply MANGO')
parser.add_argument('--mng_n_branches', type=int, default=4,
                    help='number of barnches at each node in MANGO (default: 4)')
parser.add_argument('--mng_init_len', type=int, default=16,
                    help='initial length of the masks in MANGO (default: 16)')
parser.add_argument('--n_workers', type=int, default=2,
                    help='number of workers')
parser.add_argument('--experiment_type', type=str, default='default',
                    help='default:')  # TODO

args = parser.parse_args()

args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

cudnn.benchmark = True  # Should make training should go faster for large models

torch.manual_seed(args.seed)

if args.device == 'cuda:0':   ##################?????TODO
    torch.cuda.manual_seed(args.seed)

print(args)

# test_id = args.dataset + '_' + args.model TODO

#### LOADING DATA ####
trainloader, testloader, num_classes = set_data(args)

#### TRAIN/TEST MODEL ####
model = train(trainloader, testloader, num_classes, args, False)

if args.mango:
    #### CREATING MANGO DATA ####
    mango_trainloader, num_classes = run_mango(model, trainloader,
                                  #   load_from="data/MANGO/t_train/maskD.txt",
                                  args)

    #### TRAINING WITH MANGO DATA ####
    model = train(mango_trainloader, testloader, num_classes, args, True)
