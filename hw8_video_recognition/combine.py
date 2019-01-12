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
from helperFunctions import loadFrame

import h5py
import cv2

# from multiprocessing import Pool


def getProb(base_directory):

    # action class labels
    class_file = open(base_directory + 'ucfTrainTestlist/classInd.txt','r')
    lines = class_file.readlines()
    lines = [line.split(' ')[1].strip() for line in lines]
    class_file.close()
    class_list = np.asarray(lines)


    # testing data
    test_file = open(base_directory + 'ucfTrainTestlist/testlist01.txt','r')
    lines = test_file.readlines()
    filenames = ['./UCF-101-predictions/' + line.split(' ')[0].strip() for line in lines]
    classnames = [filename.split('/')[2] for filename in filenames]
    #test_sequence = ['./new-UCF-101-predictions-sequence/' + line.split(' ')[0].strip() for line in lines]
    y_test = [np.where(classname == class_list)[0][0] for classname in classnames]
    y_test = np.asarray(y_test)
    test_file.close()

    test = (np.asarray(filenames), y_test)

    return class_list, test


IMAGE_SIZE = 224
NUM_CLASSES = 101
batch_size = 100
lr = 0.0001
num_of_epochs = 10

data_directory = '/projects/training/bauh/AR/'
class_list, test = getProb(base_directory = data_directory)

acc_top1 = 0.0
acc_top5 = 0.0
acc_top10 = 0.0
confusion_matrix = np.zeros((NUM_CLASSES,NUM_CLASSES),dtype=np.float32)

mean = np.asarray([0.485, 0.456, 0.406],np.float32)
std = np.asarray([0.229, 0.224, 0.225],np.float32)

for i in range(len(test[0])):

    t1 = time.time()

    name_single = test[0][i]
    name_sequence = test[0][i].replace('UCF-101-predictions', 'new-UCF-101-predictions-sequence')
    name_single = name_single.replace('.avi','.hdf5')
    name_sequence = name_sequence.replace('.avi','.hdf5')

    prob_single = h5py.File(name_single, 'r')
    prob_single = prob_single["predictions"].value

    prob_sequence = h5py.File(name_sequence, 'r')
    prob_sequence = prob_sequence["predictions"].value

    # softmax
    for j in range(prob_single.shape[0]):
        prob_single[j] = np.exp(prob_single[j]) / np.sum(np.exp(prob_single[j]))

    for j in range(prob_sequence.shape[0]):
        prob_sequence[j] = np.exp(prob_sequence[j]) / np.sum(np.exp(prob_sequence[j]))

    prediction = (np.sum(np.log(prob_single), axis=0) + np.sum(np.log(prob_sequence), axis=0))/2

    argsort_pred = np.argsort(-prediction)[0:10]

    label = test[1][i] 
    confusion_matrix[label, argsort_pred[0]] += 1
    if (label == argsort_pred[0]):
        acc_top1 += 1.0
    if (np.any(argsort_pred[0:5] == label)):
        acc_top5 += 1.0
    if (np.any(argsort_pred[:] == label)):
        acc_top10 += 1.0

    print('i:%d t:%f (%f,%f,%f)'
          % (i, time.time() - t1, acc_top1 / (i + 1), acc_top5 / (i + 1), acc_top10 / (i + 1)))

number_of_examples = np.sum(confusion_matrix,axis=1)
for i in range(NUM_CLASSES):
    confusion_matrix[i,:] = confusion_matrix[i,:]/np.sum(confusion_matrix[i,:])

results = np.diag(confusion_matrix)
indices = np.argsort(results)

sorted_list = np.asarray(class_list)
sorted_list = sorted_list[indices]
sorted_results = results[indices]

for i in range(NUM_CLASSES):
    print(sorted_list[i],sorted_results[i],number_of_examples[indices[i]])

np.save('combine_confusion_matrix.npy',confusion_matrix)
