import torch 
import torchvision
import torchvision.transforms as transforms

from util.MangoBox import Mango, load_from
from model import train_and_test, load_model

# INITIAL DATA
transform = transforms.Compose([transforms.ToTensor()])

batch_size = 16
num_of_epochs = 10

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

model = train_and_test(trainloader, testloader, "vanilla_model.pt",
                       num_of_epochs, save=True)
# Uncomment if need to load from file:
# model = load_model("vanilla_model.pt")

mango = Mango(model, trainloader, folder_name='t_train')
new_train = mango.create_dataset()
# Uncomment if need to load from file:
# new_train = load_from("data/MANGO/t_train/maskD.txt")

mango = Mango(model, testloader, folder_name='t_test')
new_test = mango.create_dataset()
# Uncomment if need to load from file:
# new_test = load_from("data/MANGO/t_test/maskD.txt")

print("#"*20, "\nDONE WITH SAVINGS...\n", "#"*20)

new_trainloader = torch.utils.data.DataLoader(new_train, batch_size=batch_size,
                                          shuffle=True, num_workers=2)
new_testloader = torch.utils.data.DataLoader(new_test, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

model = train_and_test(new_trainloader, new_testloader, "mango_model.pt",
                       num_of_epochs, True)

