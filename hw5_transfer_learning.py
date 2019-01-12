"""Image classification using pretrained model"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import os

file_dir = os.getcwd() + '/' + 'resnet18-5c106cde.pth'

def resnet18(pretrained=True):
    model = torchvision.models.resnet.ResNet(torchvision.models.resnet.BasicBlock, [2, 2, 2, 2])
    if pretrained:
        model.load_state_dict(torch.utils.model_zoo.load_url(file_dir, model_dir='./'))
    return model

# Build and Modify the preResNet
preResNet = resnet18()
preResNet.fc = nn.Linear(in_features=512, out_features=100, bias=True)

# Using cuda
preResNet.cuda()
preResNet = torch.nn.DataParallel(preResNet, device_ids=range(torch.cuda.device_count()))
torch.backends.cudnn.benchmark = True

# Define Loss function and optimizer
inilr = 0.001
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(preResNet.parameters(), lr=inilr)

# Define dara augmentation method
rgb_mean = (0.5, 0.5, 0.5)
rgb_std = (0.5, 0.5, 0.5)

train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(rgb_mean, rgb_std),
])

test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(rgb_mean, rgb_std)])

# load CIFAR10 data
trainset = torchvision.datasets.CIFAR100(root='~/scratch/Data4', train=True, download=True, transform=train_transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
testset = torchvision.datasets.CIFAR100(root='~/scratch/Data4', train=False, download=True, transform=test_transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)


# Create files to store training and test result
fileTraLoss = open(os.getcwd() + "/" + "trainLoss.txt", "w")
fileTraAcc = open(os.getcwd() + "/" + "trainAcc.txt", "w")
fileTesAcc = open(os.getcwd() + "/" + "testAcc.txt", "w")


# Training procedure
num_epoches = 60

# Define Upsample method
up = nn.Upsample(scale_factor=7, mode='bilinear')

for epoch in range(num_epoches):
    running_loss = 0.0
    train_accuracy = 0.0
    train_size = 0.0
    test_size = 0.0
    test_acc = 0.0

    for i, data in enumerate(trainloader, start=0):
        # input
        tra_inputs, tra_labels = data
        tra_inputs = up(tra_inputs)

        # initilize optimizer
        optimizer.zero_grad()

        # Forward
        tra_outputs = preResNet(tra_inputs)

        # backward
        tra_labels = tra_labels.to(torch.device("cuda"))
        loss = loss_function(tra_outputs, tra_labels)
        loss.backward()

        # optimize
        if (epoch >= 0):
            for group in optimizer.param_groups:
                for p in group['params']:
                    state = optimizer.state[p]
                    if ('step' in state and state['step'] >= 1024):
                        state['step'] = 1000

        optimizer.step()

        # training loss and accuracy
        running_loss += loss.item()
        train_size += tra_labels.size(0)
        _, predicted = torch.max(tra_outputs.data, 1)
        train_accuracy += (predicted == tra_labels).sum().item()

    # test accuracy
    with torch.no_grad():
        for tes_data in testloader:
            tes_image, tes_lab = tes_data
            tes_image = up(tes_image)
            tes_lab = tes_lab.to(torch.device("cuda"))
            tes_out = preResNet(tes_image)
            _, tes_predicted = torch.max(tes_out.data, 1)
            test_size += tes_lab.size(0)
            test_acc += (tes_predicted == tes_lab).sum().item()

    # Collect Result
    print('train loss [%d, %.3f]' % (epoch + 1, running_loss / train_size))
    print('train accu [%d, %.3f]' % (epoch + 1, train_accuracy / train_size))
    print('test  accu [%d, %.3f]' % (epoch + 1, test_acc / test_size))
    fileTraLoss.write(str(epoch + 1) + "," + str(round(running_loss / train_size, 4)) + "\n")
    fileTraAcc.write(str(epoch + 1) + "," + str(round(train_accuracy / train_size, 4)) + "\n")
    fileTesAcc.write(str(epoch + 1) + "," + str(round(test_acc / test_size, 4)) + "\n")

print("Training end")

# close file
fileTesAcc.close()
fileTraAcc.close()
fileTraLoss.close()

# Save model
torch.save(preResNet.state_dict(), 'erchiPreResNet12.pt')
