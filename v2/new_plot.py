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
model = ResNet18(num_classes=10).to('cuda:0')
model.load_state_dict(torch.load('models/MANGO/OrigMANGO-1ph-faster-s0.tar')['model_state_dict'])

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
activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

def test(layer_num, layer_size):
    # Get the activations of the layer of interest
    activations_avg =[0]*layer_size
    # activations_avg3=[]
    # activations_avg4=[]
    model.eval()
    for images, labels in testloader:
        images, labels = images.to('cuda:0'), labels.to('cuda:0')
        #get the outputs of the forward pass
        o= model(images)
        output = o[layer_num].to('cpu')
        #extract the activations of the neurons in the layer of interest
        for i in range(len(images)):
            for j in range(layer_size):
                activations_avg[j] = activations_avg[j] + output[i][j].sum()

    #average the activations
    activations_avg = np.mean(activations_avg, axis=0)
    print(activations_avg)
    print(activations_avg.shape)

def plot_activations(layer_number, layer_size):
    # Get the activations of the layer of interest
    activations_avg =[0]*layer_size
    model.eval()
    for images, labels in testloader:
        images, labels = images.to('cuda:0'), labels.to('cuda:0')
        model.layer2.register_forward_hook(get_activation(layer_number))
        outputs = model(images).to('cpu')
        for i in range(len(images)):
            for j in range(len(activation[layer_number])):
                activations_avg[j] = activations_avg[j] + activation[layer_number][i][j].sum()
        # Extract the activations of the neurons in the layer of interest
        
    # divide all elements by the number of data points multiplied by the number of neurons

    activations_avg[:] = [x / (len(testset)*activation[layer_number].size(2)*activation[layer_number].size(3)) for x in activations_avg]
    #sort the activations in descending order
    activations_avg.sort(reverse=True)
    # Plot the activations in descending order
    plt.bar(range(layer_size), activations_avg)
    
    # plt.barh(np.arange(len(activations_avg)), activations_avg, color='blue')
    # plt.yticks(np.arange(len(activations_avg)), np.arange(len(activations_avg)))
    plt.savefig('new'+str(layer_number)+'.png')

def main():
    # Plot the activations of the first layer
    # test(2,128)
    # Plot the activations of the second layer
    plot_activations('layer3',256)

if __name__ == '__main__':
    main()
