import torch
from models.model_VGG import modelVGG, VGG16
from models.model_simple import train_and_test, Net

def train_VGG(trainloader, testloader, batch_size, num_of_epochs,
              load_path=''):
    if load_path:
        return torch.load(load_path)
    return modelVGG(trainloader, testloader, batch_size, num_of_epochs)

def load_model(PATH, net):
    # assert(PATH is None, "PATH is None not allowed, \
    #                           please specify a PATH")
    net.load_state_dict(torch.load(PATH))
    print("Model is loading from", "../models/" + PATH)
    return net