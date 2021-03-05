from numpy.lib import type_check
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim

from pathlib import Path

from tqdm import tqdm

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 12, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(12, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)

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
    print("Using", device)

    net = Net()
    net.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

    for epoch in range(num_of_epochs): 
        running_loss = 0.0
        correct = 0.0
        total = 0.0

        progress_bar = tqdm(trainloader)
        for i, (inputs, labels) in enumerate(progress_bar, 0):
            progress_bar.set_description('Epoch ' + str(epoch + 1))
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs).to(device)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Calculate running average of accuracy and loss
            outputs = torch.max(outputs.data, 1)[1]
            total += labels.size(0)
            correct += (outputs == labels.data).sum().item()
            accuracy = correct / total

            running_loss += loss.item()

            # print statistics
            progress_bar.set_postfix(
                loss='%.3f' % (running_loss / (i + 1)),
                acc='%.3f' % accuracy)

        correct = 0.
        total = 0.
        net.eval()
        with torch.no_grad():
            for data in testloader:
                images, labels = data[0].to(device), data[1].to(device)
                outputs = net(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        tqdm.write('test_acc: %.3f' % (100 * correct / total))
        net.train()

    if save:
        assert(PATH is None, "PATH is None not allowed, \
                                  please specify a PATH")
        assert(PATH[-3:] == '.pt', "PATH extension is wrong \
                                    please use '.pt' to save the model")
        print("Model saving to", "models/" + PATH)
        print("models/ created...")
        Path("models/").mkdir(parents=True, exist_ok=True)
        torch.save(net.state_dict(), "models/" + PATH)

    # row = {'epoch': str(epoch), 'train_acc': str(accuracy), 'test_acc': str(test_acc)} 
    # csv_logger.writerow(row)

    return net

def load_model(PATH):
    assert(PATH is None, "PATH is None not allowed, \
                              please specify a PATH")
    net = Net()
    net.load_state_dict(torch.load("models/" + PATH))
    print("Model is loading from", "models/" + PATH)
    return net