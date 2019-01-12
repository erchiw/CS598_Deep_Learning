"""CNN from scratch"""

import numpy as np
import h5py
import time
import copy
import sys
from random import randint

##load MNIST data
MNIST_data = h5py.File('F:\CS598\MNISTdata.hdf5', 'r')
x_train = np.float32(MNIST_data['x_train'][:])
y_train = np.int32(np.array(MNIST_data['y_train'][:, 0]))
x_test = np.float32(MNIST_data['x_test'][:])
y_test = np.int32(np.array(MNIST_data['y_test'][:, 0]))
MNIST_data.close()

##Inisilation
num_inputs = 28*28
d = 28
num_outputs = 10

# filters
ky = 3 
kx = 3
c = 10

model = {}
model['W'] = np.random.randn(d - ky + 1, d - kx + 1, c, num_outputs) /  np.sqrt(num_outputs * (d - ky + 1) * (d - kx + 1))
model['k'] = np.random.randn(ky, kx, c)/ np.sqrt(kx * ky)
model['b'] = np.zeros(num_outputs)
model_grads = copy.deepcopy(model)

def convolution(x, k, kx, ky):
    """2d Convolution function
    
    Arguments:
    x -- input data, column vector
    k -- filter, its dimension is ky*kx
    c -- number of channels
    (default stride is 1)
    """
    d1 = x.shape[0]
    d2 = x.shape[1]
    dim1 = d1 - ky + 1
    dim2 = d2 - kx + 1
    conv = np.zeros(shape=(dim1, dim2))

    for i in range(dim1):
        for j in range(dim2):
            conv[i, j] = np.sum(x[i:(i+ky), j:(j+kx)]*k[:, :])
    return conv

def softmax_function(z):
    ZZ = np.exp(z)/np.sum(np.exp(z))
    return ZZ

def relu_function(z):
    relu = np.maximum(z,0)
    return relu

def de_relu_function(z):
    """derivative of relu function"""
    de_relu = copy.deepcopy(z)
    de_relu[de_relu >= 0] = 1
    de_relu[de_relu < 0] = 0
    return de_relu

def backward(y, z, c, model, model_grads):
    p[y] = p[y] - 1
    model_grads['b'] = p  
    delta = np.zeros(shape=(d - ky + 1, d - kx + 1, c))

    for i in range(num_outputs):
        delta = delta + p[i] * model['W'][:, :, :, i]

    for i in range(c):
        t1 = de_relu_function(z[:, :, :]) * delta[:, :, :]
        model_grads['k'][:, :, i] = convolution(x, t1[:, :, i], t1.shape[0], t1.shape[1])

    for i in range(num_outputs):
        model_grads['W'][:, :, :, i] = p[i]*h

    return model_grads


# training
time1 = time.time()
LR = .001
num_epochs = 10
train_accuracy = np.zeros(num_epochs)
test_accuracy = np.zeros(num_epochs)
u = np.zeros(num_outputs)

for epochs in range(num_epochs):
    # Learning rate schedule
    if (epochs > 4):
        LR = 0.0001
    if (epochs > 8):
        LR = 0.00001
    total_correct = 0

    for n in range(len(x_train)):
        n_random = randint(0, len(x_train) - 1)
        y = y_train[n_random]
        x = x_train[n_random][:]
        x_matrix = x.reshape((d, d))
        # Forward Step
        z = np.zeros(shape=(d-ky+1, d-kx+1, c))
        for i in range(c):
            z[:, :, i] = convolution(x_matrix, model['k'][:, :, i], kx, ky)

        h = relu_function(z)
        # compute u
        for i in range(num_outputs):
            u[i] = np.sum(model['W'][:, :, :, i] * h) + model['b'][i]

        p = softmax_function(u)

        prediction = np.argmax(p)
        if (prediction == y):
            total_correct += 1

        model_grads = backward(y, z, c, model, model_grads)
        model['b'] = model['b'] - LR * model_grads['b']
        model['W'] = model['W'] - LR * model_grads['W']
        model['k'] = model['k'] - LR * model_grads['k']
    train_accuracy[epochs] = total_correct / np.float(len(x_train))
    print(train_accuracy[epochs])

# test data
total_correct = 0
for n in range(len(x_test)):
    y = y_test[n]
    x = x_test[n][:]
    x_matrix = x.reshape((d, d))

    # Forward Step
    z = np.zeros(shape=(d - ky + 1, d - kx + 1, c))
    for i in range(c):
        z[:, :, i] = convolution(x_matrix, model['k'][:, :, i], kx, ky)

    h = relu_function(z)
    # compute u
    for i in range(num_outputs):
        u[i] = np.sum(model['W'][:, :, :, i] * h) + model['b'][i]

    p = softmax_function(u)

    prediction = np.argmax(p)
    if (prediction == y):
        total_correct += 1

test_accuracy = np.array([total_correct / np.float(len(x_test))])
print(test_accuracy)

# output results
file_name1 = sys.path[0] + "\\train_accuracy.csv"
file_name2 = sys.path[0] + "\\test_accuracy.csv"
np.savetxt(file_name1, train_accuracy, delimiter=',')
np.savetxt(file_name2, test_accuracy, delimiter=',')
