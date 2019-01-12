import os
import torch
import torchvision

import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

from torch.autograd import Variable


# Define Discriminator
class MyDiscriminator(nn.Module):

    def __init__(self):
        super(MyDiscriminator, self).__init__()
        num_feature = 128
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=num_feature,
                               kernel_size=3, padding=1, stride=1)
        self.conv2 = nn.Conv2d(in_channels=num_feature, out_channels=num_feature,
                               kernel_size=3, padding=1, stride=2)
        self.conv3 = nn.Conv2d(in_channels=num_feature, out_channels=num_feature,
                               kernel_size=3, padding=1, stride=1)
        self.conv4 = nn.Conv2d(in_channels=num_feature, out_channels=num_feature,
                               kernel_size=3, padding=1, stride=2)
        self.conv5 = nn.Conv2d(in_channels=num_feature, out_channels=num_feature,
                               kernel_size=3, padding=1, stride=1)
        self.conv6 = nn.Conv2d(in_channels=num_feature, out_channels=num_feature,
                               kernel_size=3, padding=1, stride=1)
        self.conv7 = nn.Conv2d(in_channels=num_feature, out_channels=num_feature,
                               kernel_size=3, padding=1, stride=1)
        self.conv8 = nn.Conv2d(in_channels=num_feature, out_channels=num_feature,
                               kernel_size=3, padding=1, stride=2)

        self.ln1 = nn.LayerNorm([num_feature, 32, 32])
        self.ln2 = nn.LayerNorm([num_feature, 16, 16])
        self.ln3 = nn.LayerNorm([num_feature, 16, 16])
        self.ln4 = nn.LayerNorm([num_feature, 8, 8])
        self.ln5 = nn.LayerNorm([num_feature, 8, 8])
        self.ln6 = nn.LayerNorm([num_feature, 8, 8])
        self.ln7 = nn.LayerNorm([num_feature, 8, 8])
        self.ln8 = nn.LayerNorm([num_feature, 4, 4])

        self.fc1 = nn.Linear(in_features=num_feature, out_features=1)
        self.fc10 = nn.Linear(in_features=num_feature, out_features=10)

    def forward(self, x):
        out = F.leaky_relu(self.ln1(self.conv1(x)))
        out = F.leaky_relu(self.ln2(self.conv2(out)))
        out = F.leaky_relu(self.ln3(self.conv3(out)))
        out = F.leaky_relu(self.ln4(self.conv4(out)))
        out = F.leaky_relu(self.ln5(self.conv5(out)))
        out = F.leaky_relu(self.ln6(self.conv6(out)))
        out = F.leaky_relu(self.ln7(self.conv7(out)))
        out = F.leaky_relu(self.ln8(self.conv8(out)))
        out = F.max_pool2d(out, kernel_size=4, padding=0, stride=4)
        out = out.view(-1, 128*1*1)
        out1 = self.fc1(out)
        out10 = self.fc10(out)
        outlist = [out1, out10]
        return outlist


# Data loader
transform_train = transforms.Compose([
    transforms.RandomResizedCrop(32, scale=(0.7, 1.0), ratio=(1.0,1.0)),
    transforms.ColorJitter(
            brightness=0.1*torch.randn(1),
            contrast=0.1*torch.randn(1),
            saturation=0.1*torch.randn(1),
            hue=0.1*torch.randn(1)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

transform_test = transforms.Compose([
    transforms.CenterCrop(32),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

batch_size = 128

trainset = torchvision.datasets.CIFAR10(root='./', train=True, download=True,
                                        transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=8)

testset = torchvision.datasets.CIFAR10(root='./', train=False, download=True,
                                       transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=8)

# train without generator

# Build discriminator
model_1 = MyDiscriminator()

# Use cuda
model_1.cuda()
model_1 = torch.nn.DataParallel(model_1, device_ids=range(torch.cuda.device_count()))
torch.backends.cudnn.benchmark = True

# Define criterion and optimizer
learning_rate = 0.0001
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model_1.parameters(), lr=learning_rate)

# Create files to store training and test result
fileTraLoss = open(os.getcwd() + "/" + "trainLoss.txt", "w")
fileTraAcc = open(os.getcwd() + "/" + "trainAcc.txt", "w")
fileTesAcc = open(os.getcwd() + "/" + "testAcc.txt", "w")

# Train
num_epoch = 100

for epoch in range(num_epoch):
    running_loss = 0.0
    train_accuracy = 0.0
    train_size = 0.0
    test_size = 0.0
    test_acc = 0.0

    if (epoch == 50):
        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rate / 10.0
    if (epoch == 75):
        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rate / 100.0

    for batch_idx, (X_train_batch, Y_train_batch) in enumerate(trainloader, 0):

        # input
        if (Y_train_batch.shape[0] < batch_size):
            continue
        X_train_batch = Variable(X_train_batch).cuda()
        Y_train_batch = Variable(Y_train_batch).cuda()

        # Forward
        #print(X_train_batch.size())
        output = model_1(X_train_batch)
        output = output[1]
        train_loss = criterion(output, Y_train_batch)

        # Backward
        optimizer.zero_grad()
        train_loss.backward()

        if (epoch >= 0):
            for group in optimizer.param_groups:
                for p in group['params']:
                    state = optimizer.state[p]
                    if ('step' in state and state['step'] >= 1024):
                        state['step'] = 1000
        optimizer.step()

        # training loss and accuracy
        running_loss += train_loss.item()
        train_size += Y_train_batch.size(0)
        _, predicted = torch.max(output.data, 1)
        train_accuracy += (predicted == Y_train_batch).sum().item()

    # test Acc
    with torch.no_grad():
        for (tes_image, tes_lab) in testloader:
            tes_image = Variable(tes_image).cuda()
            tes_lab = Variable(tes_lab).cuda()
            tes_out = model_1(tes_image)[1]
            _, tes_predicted = torch.max(tes_out.data, 1)
            test_size += tes_lab.size(0)
            test_acc += (tes_predicted == tes_lab).sum().item()

    # Collect result
    print('train loss [%d, %.3f]' % (epoch + 1, running_loss / train_size))
    print('train accu [%d, %.3f]' % (epoch + 1, train_accuracy / train_size))
    print('test  accu [%d, %.3f]' % (epoch + 1, test_acc / test_size))
    fileTraLoss.write(str(epoch+1)+","+str(round(running_loss / train_size, 4))+"\n")
    fileTraAcc.write(str(epoch+1)+","+str(round(train_accuracy / train_size, 4))+"\n")
    fileTesAcc.write(str(epoch+1)+","+str(round(test_acc / test_size, 4))+"\n")

torch.save(model_1, 'cifar10.model')

# close file
fileTesAcc.close()
fileTraAcc.close()
fileTraLoss.close()

