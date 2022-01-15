# Loading the model, extracting the activations
# collecting them in a list and plotting them
# in descending order
# 
# Author: Nikoo Aghaei
# Date: 2021-10-19

import numpy as np
import matplotlib.pyplot as plt
from numpy.core.fromnumeric import shape, size

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import pickle

from models.resnet import ResNet18
from models.resnetF import ResNetF18

# Load the model
mango_model = ResNet18(num_classes=10).to('cuda:0')
mango_model.load_state_dict(torch.load('models/MANGO/OrigMANGO-1ph-faster-s0.tar')['model_state_dict'])
cutout_model = ResNet18(num_classes=10).to('cuda:0')
cutout_model.load_state_dict(torch.load('models/MANGO/cutout-c10-s0.tar'))
base_model = ResNet18(num_classes=10).to('cuda:0')
base_model.load_state_dict(torch.load('models/OrigMng/base_c10_res18_s1.tar')['model_state_dict'])

cutout_normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                        std=[x / 255.0 for x in [63.0, 62.1, 66.7]])

test_transform = transforms.Compose([
        transforms.ToTensor(), cutout_normalize])
testset = torchvision.datasets.CIFAR10(root='../data', train=False,
											download=True, transform=test_transform)

testloader = torch.utils.data.DataLoader(testset, batch_size=128,
											shuffle=False, num_workers=2)

    
# Each bar on the graph represents the activation of a neuron
# in the given layer of the model (i.e. Residual Block 1)
# The number of bars is equal to the number of neurons in the layer
# 
# Basically;
#   - For each data point in the test set,
#   - Feed the data point to the model
#   - Extract the activations of the neurons in the layer of interest
#   - Add the activations to a list and average them
#       (i.e. for 'neuron 0', activation_list[0] += activation for data point)
#   - Plot the list in descending order


# def test(layer_num, layer_size):
#     # Get the activations of the layer of interest
#     activations_avg =[0]*layer_size
#     # activations_avg3=[]
#     # activations_avg4=[]
#     mango_model.eval()
#     for images, labels in testloader:
#         images, labels = images.to('cuda:0'), labels.to('cuda:0')
#         #get the mng_ of the forward pass
#         o= mango_model(images)
#         output = o[layer_num].to('cpu')
#         #extract the activations of the neurons in the layer of interest
#         for i in range(len(images)):
#             for j in range(layer_size):
#                 activations_avg[j] = activations_avg[j] + output[i][j].sum()

#     #average the activations
#     activations_avg = np.mean(activations_avg, axis=0)
#     print(activations_avg)
#     print(activations_avg.shape)
mango_activation = {}
cutout_activation = {}
base_activation = {}
def get_mng_activation(name):
    def hook(model, input, output):
        mango_activation[name] = output.detach()
    return hook
def get_cutout_activation(name):
    def hook(model, input, output):
        cutout_activation[name] = output.detach()
    return hook
def get_base_activation(name):
    def hook(model, input, output):
        base_activation[name] = output.detach()
    return hook

def plot_activations(layer_name, layer_size):
    # Get the activations of the layer of interest
    mng_activations_avg =[0]*layer_size
    cutout_activations_avg =[0]*layer_size
    base_activations_avg =[0]*layer_size

    mango_model.eval()
    cutout_model.eval()
    base_model.eval()
    for images, labels in testloader:
        images, labels = images.to('cuda:0'), labels.to('cuda:0')
        mango_model.layer4.register_forward_hook(get_mng_activation(layer_name))
        cutout_model.layer4.register_forward_hook(get_cutout_activation(layer_name))
        base_model.layer4.register_forward_hook(get_base_activation(layer_name))
        mng_outputs = mango_model(images).to('cpu')
        cutout_outputs = cutout_model(images).to('cpu')
        base_outputs = base_model(images).to('cpu')
        for i in range(layer_size):
            for j in range(len(images)):
                # Extract the activations of the neurons in the layer of interest for each data point
                ####sum all the activations
                mng_activations_avg[i] = mng_activations_avg[i] + mango_activation[layer_name][j][i].sum()
                cutout_activations_avg[i] = cutout_activations_avg[i] + cutout_activation[layer_name][j][i].sum()
                base_activations_avg[i] = base_activations_avg[i] + base_activation[layer_name][j][i].sum()
                ####sum the max activations
                # mng_activations_avg[i] = mng_activations_avg[i] + torch.max(mango_activation[layer_name][j][i])
                # cutout_activations_avg[i] = cutout_activations_avg[i] + torch.max(cutout_activation[layer_name][j][i])
                # base_activations_avg[i] = base_activations_avg[i] + torch.max(base_activation[layer_name][j][i])
                ####count activations > 0
                # mng_activations_avg[i] = mng_activations_avg[i] + torch.count_nonzero(mango_activation[layer_name][j][i])
                # cutout_activations_avg[i] = cutout_activations_avg[i] + torch.count_nonzero(cutout_activation[layer_name][j][i])
                # base_activations_avg[i] = base_activations_avg[i] + torch.count_nonzero(base_activation[layer_name][j][i])
    # divide all elements by the number of data points multiplied by the number of neurons
    # mng_activations_avg[:] = [x / len(testset) for x in mng_activations_avg]
    # cutout_activations_avg[:] = [x / len(testset) for x in cutout_activations_avg]
    # base_activations_avg[:] = [x / len(testset) for x in base_activations_avg]
    mng_activations_avg[:] = [x / (len(testset)*mango_activation[layer_name].size(2)*mango_activation[layer_name].size(3)) for x in mng_activations_avg]
    cutout_activations_avg[:] = [x / (len(testset)*cutout_activation[layer_name].size(2)*cutout_activation[layer_name].size(3)) for x in cutout_activations_avg]
    base_activations_avg[:] = [x / (len(testset)*base_activation[layer_name].size(2)*base_activation[layer_name].size(3)) for x in base_activations_avg]
    
    #sort the activations in descending order
    mng_activations_avg.sort(reverse=True)
    cutout_activations_avg.sort(reverse=True)
    base_activations_avg.sort(reverse=True)
    
    plt.figure(figsize=(10,10))
    # Plot the activations in stacked bar plots
    plt.bar(np.arange(layer_size), mng_activations_avg, color='r', label='MANGO')
    plt.bar(np.arange(layer_size), cutout_activations_avg, color='b', label='cutout')
    plt.bar(np.arange(layer_size), base_activations_avg, color='g', label='base')
    ####separate bar plots next to each other
    # plt.bar(np.arange(layer_size), base_activations_avg, color='g', label='base', width= 0.5)
    # plt.bar(np.arange(layer_size)+0.33, cutout_activations_avg, color='b', label='cutout', width= 0.5)
    # plt.bar(np.arange(layer_size)+1, mng_activations_avg, color='r', label='MANGO', width= 0.5)
    ####
    plt.legend()
    plt.xlabel('Neuron')
    plt.ylabel('Average of activation')
    plt.title('Activations of the '+ layer_name)
    plt.savefig('plots/'+layer_name+'sum2.svg')

def main():
    # Plot the activations of the second layer
    plot_activations('layer4',512)

if __name__ == '__main__':
    main()
