import numpy as np
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
#from torchvision import datasets, transforms
from torch.autograd import Variable
import torch.distributed as dist
import torchvision

from helperFunctions import getUCF101
from helperFunctions import loadSequence
import resnet_3d

import h5py
import cv2

from multiprocessing import Pool


IMAGE_SIZE = 224
NUM_CLASSES = 101
batch_size = 16
lr = 0.0001
num_of_epochs = 10


data_directory = '/projects/training/bauh/AR/'
class_list, train, test = getUCF101(base_directory = data_directory)


model =  resnet_3d.resnet50(sample_size=IMAGE_SIZE, sample_duration=16)
pretrained = torch.load(data_directory + 'resnet-50-kinetics.pth')
keys = [k for k,v in pretrained['state_dict'].items()]
pretrained_state_dict = {k[7:]: v.cpu() for k, v in pretrained['state_dict'].items()}
model.load_state_dict(pretrained_state_dict)
model.fc = nn.Linear(model.fc.weight.shape[1],NUM_CLASSES)


for param in model.parameters():
    param.requires_grad_(False)

# for param in model.conv1.parameters():
#     param.requires_grad_(True)
# for param in model.bn1.parameters():
#     param.requires_grad_(True)
# for param in model.layer1.parameters():
#     param.requires_grad_(True)
# for param in model.layer2.parameters():
#     param.requires_grad_(True)
# for param in model.layer3.parameters():
#     param.requires_grad_(True)
for param in model.layer4[0].parameters():
    param.requires_grad_(True)
for param in model.fc.parameters():
    param.requires_grad_(True)

params = []
# for param in model.conv1.parameters():
#     params.append(param)
# for param in model.bn1.parameters():
#     params.append(param)
# for param in model.layer1.parameters():
#     params.append(param)
# for param in model.layer2.parameters():
#     params.append(param)
# for param in model.layer3.parameters():
#     params.append(param)
for param in model.layer4[0].parameters():
    params.append(param)
for param in model.fc.parameters():
    params.append(param)


model.cuda()

optimizer = optim.Adam(params,lr=lr)

criterion = nn.CrossEntropyLoss()

pool_threads = Pool(8,maxtasksperchild=200)

for epoch in range(0, num_of_epochs):

    ###### TRAIN
    train_accu = []
    model.train()
    random_indices = np.random.permutation(len(train[0]))
    start_time = time.time()
    for i in range(0, len(train[0]) - batch_size, batch_size):

        augment = True
        video_list = [(train[0][k], augment)
                      for k in random_indices[i:(batch_size + i)]]
        data = pool_threads.map(loadSequence, video_list)
         
    
        next_batch = 0
        for video in data:
            if video.size == 0:  # there was an exception, skip this
                next_batch = 1
        if (next_batch == 1):
            continue

        x = np.asarray(data, dtype=np.float32)
        x = Variable(torch.FloatTensor(x), requires_grad=False).cuda().contiguous()
          
        y = train[1][random_indices[i:(batch_size + i)]]
        y = torch.from_numpy(y).cuda()
        
        print("x size is", x.shape)
        print("y size is", y.shape)
        with torch.no_grad():
            h = model.conv1(x)
            h = model.bn1(h)
            h = model.relu(h)
            h = model.maxpool(h)

            h = model.layer1(h)
            h = model.layer2(h)
            h = model.layer3(h)
        h = model.layer4[0](h)

        h = model.avgpool(h)

        h = h.view(h.size(0), -1)
        output = model.fc(h)

        # output = model(x)

        loss = criterion(output, y)
        optimizer.zero_grad()

        loss.backward()
        optimizer.step()

        prediction = output.data.max(1)[1]
        accuracy = (float(prediction.eq(y.data).sum()) / float(batch_size)) * 100.0
        if (epoch == 0):
            print(i, accuracy)
        train_accu.append(accuracy)
    accuracy_epoch = np.mean(train_accu)
    print(epoch, accuracy_epoch, time.time() - start_time)

    ##### TEST
    model.eval()
    test_accu = []
    random_indices = np.random.permutation(len(test[0]))
    t1 = time.time()
    for i in range(0, len(test[0]) - batch_size, batch_size):
        augment = False
        video_list = [(test[0][k], augment)
                      for k in random_indices[i:(batch_size + i)]]
        data = pool_threads.map(loadSequence, video_list)

        next_batch = 0
        for video in data:
            if video.size == 0:  # there was an exception, skip this batch
                next_batch = 1
        if (next_batch == 1):
            continue

        x = np.asarray(data, dtype=np.float32)
        x = Variable(torch.FloatTensor(x)).cuda().contiguous()

        y = test[1][random_indices[i:(batch_size + i)]]
        y = torch.from_numpy(y).cuda()

        # with torch.no_grad():
        #     output = model(x)
        with torch.no_grad():
            h = model.conv1(x)
            h = model.bn1(h)
            h = model.relu(h)
            h = model.maxpool(h)

            h = model.layer1(h)
            h = model.layer2(h)
            h = model.layer3(h)
            h = model.layer4[0](h)
            # h = model.layer4[1](h)

            h = model.avgpool(h)

            h = h.view(h.size(0), -1)
            output = model.fc(h)

        prediction = output.data.max(1)[1]
        accuracy = (float(prediction.eq(y.data).sum()) / float(batch_size)) * 100.0
        test_accu.append(accuracy)
        accuracy_test = np.mean(test_accu)

    print('Testing', accuracy_test, time.time() - t1)

torch.save(model, '3d_resnet.model')
pool_threads.close()
pool_threads.terminate()

