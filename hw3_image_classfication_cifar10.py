"""Using CNN to do classification on CIFAR10"""

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.optim as optim
import os

# Open file to store train and test error
nameTra = os.getcwd() + "/" + "trainErr.txt"
nameTes = os.getcwd() + "/" + "testErr.txt"
trainErr = open(nameTra, "w")
testErr = open(nameTes, "w")

# Define dara augmentation method
rgb_mean = (0.5, 0.5, 0.5)
rgb_std = (0.5, 0.5, 0.5)

train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(p=0.2),
    transforms.ToTensor(),
    transforms.Normalize(rgb_mean, rgb_std),
])

test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(rgb_mean, rgb_std)])

# load CIFAR10 data
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=False, transform=train_transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=False, transform=test_transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Define Model Structure
class CNNet(nn.Module):
    def __init__(self):
        super(CNNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(4,4), stride=(1,1), padding=2)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(4,4), stride=(1,1), padding=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(4,4), stride=(1,1), padding=2)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(4,4), stride=(1,1), padding=2)
        self.conv5 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(4,4), stride=(1,1), padding=2)
        self.conv6 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3,3), stride=(1,1), padding=0)
        self.conv7 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3,3), stride=(1,1), padding=0)
        self.conv8 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3,3), stride=(1,1), padding=0)
        self.fc1 = nn.Linear(4*4*64, 500)
        self.fc2 = nn.Linear(500, 10)

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(64)

        self.drop1 = nn.Dropout(p=0.1)
        self.drop2 = nn.Dropout(p=0.2)
        self.drop3 = nn.Dropout(p=0.4)
        self.drop4 = nn.Dropout(p=0.5)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = self.drop1(x)
        x = F.relu(self.bn2(self.conv3(x)))
        x = F.relu(self.conv4(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.bn3(self.conv5(x)))
        x = F.relu(self.conv6(x))
        x = self.drop3(x)
        x = F.relu(self.bn4(self.conv7(x)))
        x = F.relu(self.bn5(self.conv8(x)))
        x = self.drop4(x)
        x = x.view(-1, 4*4*64)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Create Model
net = CNNet()

# Using Cuda
net.cuda()
net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
torch.backends.cudnn.benchmark = True

# Define Loss function and optimizer
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

# train the network
num_epoch = 20

for epoch in range(num_epoch):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        #input       
        inputs, labels = data
        
        #initiliza optimizer
        optimizer.zero_grad()
        
        #forward, backward and optimize
        outputs = net(inputs)
        labels = labels.to(torch.device("cuda"))
        loss = loss_function(outputs, labels)
        loss.backward()

        if (epoch >= 0):
            for group in optimizer.param_groups:
                for p in group['params']:
                    state = optimizer.state[p]
                    if ('step' in state and state['step'] >= 1024):
                        state['step'] = 1000

        optimizer.step()

        #print
        running_loss += loss.item()
        if i % 2000 == 1999:  # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            trainErr.write(str(running_loss/2000)+",")
            running_loss = 0.0

print("training finish")

# Test model
correct = 0.0
alltest = 0.0

with torch.no_grad():
    for data in testloader:
        img, lab = data
        lab = lab.to(torch.device("cuda"))
        out = net(img)
        _, predicted = torch.max(out.data, 1)
        alltest += lab.size(0)
        correct += (predicted == lab).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / alltest))

testErr.write(str(100 * correct / alltest)+",")

print("all finish")

# Save model
torch.save(net.state_dict(), 'cnn_erchi.pt')

# Close File
testErr.close()
trainErr.close()
