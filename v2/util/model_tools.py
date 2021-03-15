import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR

from tqdm import tqdm

from pathlib import Path

from models.model_base import Base
from models.resnet import ResNet18
from models.wide_resnet import WideResNet
from models.model_VGG import VGG16

def train(trainloader, testloader, num_classes, logger, args, is_mango):
    if args.first_model_load_path:
        return torch.load(args.first_model_load_path)
    model, optimizer, scheduler = _make_model(num_classes, args, is_mango)
    return _run_epochs(trainloader, testloader, model, optimizer, scheduler, logger, args, is_mango)

def _make_model(num_classes, args, is_mango):
    if is_mango:
        model = args.mng_model
    else:
        model = args.first_model
    # Creating the model
    if model == 'basemodel':
        cnn = Base()
    elif model == 'resnet18':
        cnn = ResNet18(num_classes=num_classes)
    elif model == 'wideresnet':
        cnn = WideResNet(depth=28, num_classes=num_classes, widen_factor=10,
                        dropRate=0.3)
    elif model == 'vgg16':
        cnn = VGG16(n_classes=num_classes)
    
    # Loss, Optimizer & Scheduler
    cnn = cnn.to(args.device)
    if model == 'vgg16':
        optimizer = optim.Adam(cnn.parameters(), lr=args.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    else:
        optimizer = optim.SGD(cnn.parameters(), lr=args.learning_rate,
                                    momentum=0.9, nesterov=True, weight_decay=5e-4)
        scheduler = MultiStepLR(optimizer, milestones=[60, 120, 160], gamma=0.2)
    return cnn, optimizer, scheduler

def _run_epochs(trainLoader, testLoader, model, optimizer, scheduler, csv_logger, args, is_mango):
    if is_mango:
        model_name = args.mng_model
    else:
        model_name = args.first_model

    criterion = nn.CrossEntropyLoss().to(args.device)

    # Train and test the model
    for epoch in range(args.n_epochs):
        correct = 0.
        total = 0.
        avg_loss = 0.

        progress_bar = tqdm(trainLoader)
        for i, (images, labels) in enumerate(progress_bar):
            progress_bar.set_description('Epoch ' + str(epoch + 1))

            # get the inputs; data is a list of [images, labels]
            images, labels = images.to(args.device), labels.to(args.device)

            # Forward + Backward + Optimize
            optimizer.zero_grad()
            outputs = model(images).to(args.device)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            # Calculate running average of accuracy and loss
            outputs = torch.max(outputs.data, 1)[1]
            total += labels.size(0)
            correct += (outputs == labels.data).sum().item()
            accuracy = correct / total

            avg_loss += loss.item()

            # print statistics
            progress_bar.set_postfix(
                loss='%.3f' % loss.data,
                avg_loss='%.3f' % (avg_loss / (i + 1)),
                train_acc='%.3f' % accuracy)

        if model_name == 'vgg16':
            scheduler.step(avg_loss)
        elif model_name == 'resnet18' or model_name == 'wideresnet':
            scheduler.step()
        # No scheduler.step for basemodel

        # Test the model after each epoch
        test_acc = test(model, testLoader, args)
        tqdm.write('test_acc: %.3f' % test_acc)
    
        # Save output log
        row = {'epoch': str(epoch + 1), 'train_acc': "  " + str('%.3f' % accuracy), 'test_acc': "  " + str('%.3f' % test_acc)} 
        csv_logger.writerow(row)

    # Save the Trained Model
    if args.save_models:
        # assert args.model_save_path[-3:] == '.pt', "PATH extension is wrong \
        #                             please use '.pt' to save the model"
        if is_mango:
            print("Model saving to", "models/MANGO/")
            print("models/MANGO created...")
            Path("models/MANGO/").mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), "models/MANGO/" + args.experiment_type + ".pt")
        else:
            print("Model saving to", "models/")
            print("models/ created...")
            Path("models/").mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), "models/" + args.experiment_type + ".pt")
    return model

def test(model, loader, args):
    model.eval()
    correct = 0.
    total = 0.

    for images, labels in loader:
        images, labels = images.to(args.device), labels.to(args.device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    model.train()

    return 100 * correct / total