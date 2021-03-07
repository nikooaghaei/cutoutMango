import torch
import torchvision
import torchvision.transforms as transforms

def set_data_CIFAR10(batch_size, n_workers=2):
    transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean = [ 0.485, 0.456, 0.406 ],
                                std  = [ 0.229, 0.224, 0.225 ]),
            ])

    trainset = torchvision.datasets.CIFAR10(root='../data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=n_workers)

    testset = torchvision.datasets.CIFAR10(root='../data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=n_workers)

    return trainloader, testloader
    