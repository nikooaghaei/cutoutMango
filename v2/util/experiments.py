import argparse
from pathlib import Path

import torch

from util.data import set_data
from util.misc import CSVLogger
from util.model_tools import train_and_test
from util.MangoBox import run_mango

class Experiment():
    def __init__(self, description = 'CNN'):
        self.model_options = ['basemodel', 'resnet18', 'wideresnet', 'vgg16']
        self.dataset_options = ['cifar10', 'cifar100']
        parser = argparse.ArgumentParser(description)
        self.args = self._set_args(parser)

    def _set_args(self, parser):
        parser.add_argument('--first_dataset', '-d', default='cifar10',
                            choices=self.dataset_options)
        parser.add_argument('--first_model', '-m', default='resnet18',
                            choices=self.model_options)
        parser.add_argument('--mng_dataset', '-md', default='cifar10',
                            choices=self.dataset_options)
        parser.add_argument('--mng_model', '-mm', default='resnet18',
                            choices=self.model_options)
        parser.add_argument('--first_model_load_path', '-mlp', default='',
                            help='path for both models to load from(default: None)')
        parser.add_argument('--mng_load_data_path', '-ld', default='',
                            help='path for MANGO data to load (default: None)')
        parser.add_argument('--batch_size', type=int, default=128,
                            help='input batch size for training (default: 128)')
        parser.add_argument('--n_epochs', type=int, default=1,
                            help='number of epochs to train (default: 200)')
        parser.add_argument('--learning_rate', '-lr', type=float, default=0.01,
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
                            help='random seed (default: 1)')
        parser.add_argument('--mango', action='store_true', default=False,
                            help='apply MANGO')
        parser.add_argument('--mng_n_branches', type=int, default=4,
                            help='number of barnches at each node in MANGO (default: 4)')
        parser.add_argument('--mng_init_len', type=int, default=16,
                            help='initial length of the masks in MANGO (default: 16)')
        parser.add_argument('--n_workers', type=int, default=2,
                            help='number of workers')
        parser.add_argument('--experiment_type', '-ex',type=str, default='test_ex',
                            help='Name for saving first model in models/, saving MANGO model in models/MANGO/, \
                                saving MANGO data in data/MANGO, saving csv loggers in logs/ for first phase and \
                                    logs/MANGO/ for second phase (default:test_ex)')  # TODO

        args = parser.parse_args()
        args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        return args

    def run_experiment(self):
        first_trainloader, first_testloader, trained_model = self._run_first_phase()

        if self.args.mango:
            self._run_second_phase(first_trainloader, first_testloader, trained_model)

    def _run_first_phase(self):

        #### CREATING THE DATASET ####
        trainloader, testloader, num_classes = set_data(self.args)

        print("logs/ created...")
        Path("logs/").mkdir(parents=True, exist_ok=True)
        log_filename = '../logs/' + self.args.experiment_type + '.csv'
        csv_logger = CSVLogger(args=self.args, fieldnames=['epoch', 'train_acc', 'test_acc'], filename=log_filename)

        #### TRAIN/TEST MODEL ####
        model = train_and_test(trainloader, testloader, num_classes, csv_logger, self.args, False)

        csv_logger.close()

        return trainloader, testloader, model

    def _run_second_phase(self, first_trainloader, first_testloader, pretrained_model):

        #### CREATING MANGO DATA ####
        mango_trainloader, num_classes = run_mango(pretrained_model, first_trainloader, self.args)
        
        print("logs/MANGO/ created...")
        Path("logs/MANGO/").mkdir(parents=True, exist_ok=True)
        log_filename = '../logs/MANGO/' + self.args.experiment_type + '.csv'
        mng_csv_logger = CSVLogger(args=self.args, fieldnames=['epoch', 'train_acc', 'test_acc'], filename=log_filename)

        #### TRAINING WITH MANGO DATA ####
        mng_model = train_and_test(mango_trainloader, first_testloader, num_classes, mng_csv_logger, self.args, True)

        mng_csv_logger.close()
    

class LR_Experiment(Experiment):
    def run_experiment(self, values):
        for value in values:
            self.args.learning_rate = value
            self.args.save_models = 'lr_ex_'+value+'_'+self.args.save_models
            super().run_experiment()

class Batch_Size_Experiment(Experiment):
    def run_experiment(self, sizes):
        for size in sizes:
            self.args.batch_size = size
            self.args.save_models = 'batch_ex_'+size+'_'+self.args.save_models
            super().run_experiment()

class Model_Experiment(Experiment):
    def run_experiment(self, first_models = super.model_options, mng_models = super.model_options):
        for model in first_models:
            for mng_model in mng_models:
                self.args.first_model = model
                self.args.mng_model = mng_model
                self.args.save_models = 'model_ex_'+model+'_'+mng_model+'_'+self.args.save_models
                super().run_experiment()

class Dataset_Experiment(Experiment):
    def run_experiment(self, datasets = super.dataset_options):
        for dataset in datasets:
            self.args.first_dataset = dataset
            self.args.mng_dataset = dataset
            self.args.save_models = 'dataset_ex_'+dataset+'_'+self.args.save_models
            super().run_experiment()
    

