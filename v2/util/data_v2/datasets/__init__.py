from torchvision.datasets.lsun import LSUN, LSUNClass
from torchvision.datasets.folder import ImageFolder, DatasetFolder
from torchvision.datasets.coco import CocoCaptions, CocoDetection
from .cifar import CIFAR10, CIFAR100
from torchvision.datasets.stl10 import STL10
from torchvision.datasets.mnist import MNIST, EMNIST, FashionMNIST, KMNIST, QMNIST
from torchvision.datasets.svhn import SVHN
from torchvision.datasets.phototour import PhotoTour
from torchvision.datasets.fakedata import FakeData
from torchvision.datasets.semeion import SEMEION
from torchvision.datasets.omniglot import Omniglot
from torchvision.datasets.sbu import SBU
from torchvision.datasets.flickr import Flickr8k, Flickr30k
from torchvision.datasets.voc import VOCSegmentation, VOCDetection
from torchvision.datasets.cityscapes import Cityscapes
from torchvision.datasets.imagenet import ImageNet
from torchvision.datasets.caltech import Caltech101, Caltech256
from torchvision.datasets.celeba import CelebA
from torchvision.datasets.sbd import SBDataset
from torchvision.datasets.vision import VisionDataset
from torchvision.datasets.usps import USPS
from torchvision.datasets.kinetics import Kinetics400
from torchvision.datasets.hmdb51 import HMDB51
from torchvision.datasets.ucf101 import UCF101
from torchvision.datasets.places365 import Places365

__all__ = ('LSUN', 'LSUNClass',
           'ImageFolder', 'DatasetFolder', 'FakeData',
           'CocoCaptions', 'CocoDetection',
           'CIFAR10', 'CIFAR100', 'EMNIST', 'FashionMNIST', 'QMNIST',
           'MNIST', 'KMNIST', 'STL10', 'SVHN', 'PhotoTour', 'SEMEION',
           'Omniglot', 'SBU', 'Flickr8k', 'Flickr30k',
           'VOCSegmentation', 'VOCDetection', 'Cityscapes', 'ImageNet',
           'Caltech101', 'Caltech256', 'CelebA', 'SBDataset', 'VisionDataset',
           'USPS', 'Kinetics400', 'HMDB51', 'UCF101', 'Places365')
