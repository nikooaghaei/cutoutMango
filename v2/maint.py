import argparse

import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from pathlib import Path

from util.MangoBox import load_from, run_mango
from util.model_toolst import train_and_test
from util.model_toolst import _run_epochs

from util.data import set_data
from util.data_v2.data_v2 import set_data_v2
from util.misc import CSVLogger
import multiprocessing as mp
# Increasing worker limit -seems to be necessary in some situations
# check for more: https://github.com/pytorch/pytorch/issues/973#issuecomment-346405667
import resource

if __name__ == '__main__':
    mp.set_start_method('spawn')
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
                        help='path for the first model to load from(default: None)')
    parser.add_argument('--mng_model_load_path', '-mlp2', default='',
                        help='path for the MANGO model to load from(default: None)')
    # parser.add_argument('--model_save_path', '-sp1', default='',
    #                     help='path to save both models in models/args.model_save_path (default: None)')
    parser.add_argument('--mng_load_data_path', '-ld', default='',
                        help='path for MANGO data to load (default: None)')
    parser.add_argument('--mng_save_data', '-sd', action='store_true', default=False,
                        help='save mango data in data/MANGO/args.ex (default: True)')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='input batch size for training (default: 128)')
    parser.add_argument('--n_epochs', type=int, default=200,
                        help='number of epochs to train (default: 200)')
    parser.add_argument('--mng_epochs', type=int, default=200,
                        help='number of epochs to train (default: 200)')
    parser.add_argument('--learning_rate', '-lr', type=float, default=0.1,
                        help='learning rate')
    parser.add_argument('--data_augmentation', action='store_true', default=False,
                        help='augment data by flipping and cropping')
    parser.add_argument('--save_models', action='store_true', default=False,
                        help='save both first and MANGO model in models/ and models/MANGO/ (default:F)')
    parser.add_argument('--cutout', action='store_true', default=False,
                        help='apply Cutout')
    parser.add_argument('--cutout_n_holes', type=int, default=1,
                        help='number of holes to cut out from image in Cutout')
    parser.add_argument('--cutout_len', type=int, default=16,
                        help='length of the holes in Cutout')
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed (default: 0)')
    parser.add_argument('--mango', action='store_true', default=False,
                        help='apply MANGO')

    parser.add_argument('--forced', action='store_true', default=False,
                        help='apply forcedMANGOcut')
    parser.add_argument('--gt', action='store_true', default=False,
                        help='apply MANGOcutgt')
    parser.add_argument('--origmng', action='store_true', default=False,
                        help='apply origmng')                    


    parser.add_argument('--mng_n_branches', type=int, default=4,
                        help='number of barnches at each node in MANGO (default: 4)')
    parser.add_argument('--mng_init_len', type=int, default=16,
                        help='initial length of the masks in MANGO (default: 16)')
    parser.add_argument('--n_workers', type=int, default=2,
                        help='number of workers')
    parser.add_argument('--experiment_name', '-ex',type=str, default='test_ex',
                        help='Name for saving first model in models/, saving MANGO model in models/MANGO/, \
                            saving MANGO data in data/MANGO, saving csv loggers in logs/ for first phase and \
                                logs/MANGO/ for second phase (default:test_ex)')  # TODO

    args = parser.parse_args()

    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    ###################################################################################################

    cudnn.benchmark = True  # Should make training should go faster for large models

    torch.manual_seed(args.seed)

    if args.device == 'cuda:0':   ##################?????TODO
        torch.cuda.manual_seed(args.seed)

    print(args)

    # test_id = args.dataset + '_' + args.model TODO

    #### LOADING DATA ####
    trainloader, testloader, num_classes = set_data(args)
    print("rand_logs/ created...")
    Path("rand_logs/").mkdir(parents=True, exist_ok=True)
    log_filename = 'rand_logs/' + args.experiment_name + '.csv'
    csv_logger = CSVLogger(args=args, fieldnames=['epoch', 'train_acc', 'test_acc'], filename=log_filename)

    model = train_and_test(num_classes, csv_logger, args, trainloader, testloader, False)
    if args.mango:
        #### USING MANGO AS A DATA TRANSFORM ####
        model.eval()
        mango_trainloader, mango_testloader , num_classes = set_data(args, model, is_mango = True)

        print("rand_logs/MANGO/ created...")
        Path("rand_logs/MANGO/").mkdir(parents=True, exist_ok=True)
        log_filename = 'rand_logs/MANGO/' + args.experiment_name + '.csv'
        mng_csv_logger = CSVLogger(args=args, fieldnames=['epoch', 'train_acc', 'test_acc'], filename=log_filename)

        optimizer = optim.SGD(model.parameters(), lr=args.learning_rate,
                                        momentum=0.9, nesterov=True, weight_decay=5e-4)
        scheduler = MultiStepLR(optimizer, milestones=[60, 120, 160], gamma=0.2)
        #### TRAINING WITH MANGO DATA ####
        mng_model = _run_epochs(mango_trainloader, mango_testloader, model,optimizer, scheduler, mng_csv_logger,args, True)

        mng_csv_logger.close()