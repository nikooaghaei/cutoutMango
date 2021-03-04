import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim

from pathlib import Path
import tqdm

# TODO: Add continuing training
# TODO: Fix all the -assert()

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def train_and_test(trainloader, testloader, PATH,  
                   num_of_epochs=2, save=False):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    net = Net()
    net.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    for epoch in tqdm.tqdm(range(num_of_epochs)): 
        # print(trainloader.shape)
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    print('Finished Training')

    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%' % (
        100 * correct / total))

    if save:
        assert(PATH is None, "PATH is None not allowed, \
                                  please specify a PATH")
        assert(PATH[-3:] == '.pt', "PATH extension is wrong \
                                    please use '.pt' to save the model")
        print("Model saving to", "models/" + PATH)
        print("models/ created...")
        Path("models/").mkdir(parents=True, exist_ok=True)
        torch.save(net.state_dict(), "models/" + PATH)

    return net

def load_model(PATH):
    assert(PATH is None, "PATH is None not allowed, \
                              please specify a PATH")
    net = Net()
    net.load_state_dict(torch.load("models/" + PATH))
    print("Model is loading from", "models/" + PATH)
    return net