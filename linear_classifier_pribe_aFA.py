import numpy as np
import os
import cupy as cp
import chainer
from chainer.backends import cuda
from chainer import Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions
import PIL
import keras
from keras.datasets import cifar10
import matplotlib as mpl

# Load the CIFAR-10 dataset
num_classes = 10

(x_train, t_train), (x_test, t_test) = cifar10.load_data()

# x_train = x_train.reshape(())
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

x_train = x_train.reshape(-1, 3072) / 255.
x_test = x_test.reshape(-1, 3072) / 255.

t_train = keras.utils.to_categorical(t_train, num_classes)
t_test = keras.utils.to_categorical(t_test, num_classes)

x_train = cp.array(x_train)
x_test = cp.array(x_test)
t_train = cp.array(t_train)
t_test = cp.array(t_test)


def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    if t.size == y.size:
        t = t.argmax(axis=1)

    batch_size = y.shape[0]
    return -cp.sum(cp.log(y[cp.arange(batch_size), t] + 1e-7)) / batch_size


def relu(x):
    return cp.maximum(0, x)


def relu_grad(x):
    x[x <= 0] = 0
    x[x > 0] = 1
    return x


def tanh_grad(x):
    return 1 - (cp.tanh(x) ** 2)


def softmax(x):
    if x.ndim == 2:
        x = x.T
        x = x - cp.max(x, axis=0)
        y = cp.exp(x) / cp.sum(cp.exp(x), axis=0)
        return y.T

    x = x - cp.max(x)
    return cp.exp(x) / cp.sum(cp.exp(x))


def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    if t.size == y.size:
        t = t.argmax(axis=1)

    batch_size = y.shape[0]
    return -cp.sum(cp.log(y[cp.arange(batch_size), t] + 1e-7)) / batch_size


def relu(x):
    return cp.maximum(0, x)


def relu_grad(x):
    x[x <= 0] = 0
    x[x > 0] = 1
    return x


def softmax(x):
    if x.ndim == 2:
        x = x.T
        x = x - cp.max(x, axis=0)
        y = cp.exp(x) / cp.sum(cp.exp(x), axis=0)
        return y.T

    x = x - cp.max(x)
    return cp.exp(x) / cp.sum(cp.exp(x))


hidden_unit = 1000

BP_W1 = np.loadtxt("./0822/FA_cifarW1.txt")
BP_W2 = np.loadtxt("./0822/FA_cifarW2.txt")
BP_W3 = np.loadtxt("./0822/FA_cifarW3.txt")
BP_W4 = np.loadtxt("./0822/FA_cifarW4.txt")
BP_W5 = np.loadtxt("./0822/FA_cifarW5.txt")
BP_b1 = np.loadtxt("./0822/FA_cifarb1.txt")
BP_b2 = np.loadtxt("./0822/FA_cifarb2.txt")
BP_b3 = np.loadtxt("./0822/FA_cifarb3.txt")
BP_b4 = np.loadtxt("./0822/FA_cifarb4.txt")
BP_b5 = np.loadtxt("./0822/FA_cifarb5.txt")

BP_W1 = cp.asarray(BP_W1)
BP_W2 = cp.asarray(BP_W2)
BP_W3 = cp.asarray(BP_W3)
BP_W4 = cp.asarray(BP_W4)
BP_W5 = cp.asarray(BP_W5)
BP_b1 = cp.asarray(BP_b1)
BP_b2 = cp.asarray(BP_b2)
BP_b3 = cp.asarray(BP_b3)
BP_b4 = cp.asarray(BP_b4)
BP_b5 = cp.asarray(BP_b5)

train_size = x_train.shape[0]
batch_size = 100
iter_per_epoch = 100
print("Linear classifier probe: input")
weight_init_std = 0.03
W_lin = weight_init_std * cp.random.randn(3072, 10)
b_lin = cp.zeros(10)
alpha = 0.01
for i in range(100000):
    batch_mask = cp.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]
    # h = cp.dot(x_batch,FA_W1) + FA_b1
    # h = cp.tanh(h)
    output = softmax(cp.dot(x_batch, W_lin) + b_lin)
    delta = (output - t_batch) / batch_size
    delta_W_lin = cp.dot(x_batch.T, delta)
    delta_b_lin = cp.dot(cp.ones(batch_size), delta)
    W_lin -= alpha * delta_W_lin
    b_lin -= alpha * delta_b_lin
    if i % 100 == 0:
        # h = cp.dot(x_test, FA_W1) + FA_b1
        # h = cp.tanh(h)
        output = softmax(cp.dot(x_test, W_lin) + b_lin)
        y = cp.argmax(output, axis=1)
        t = cp.argmax(t_test, axis=1)
        accuracy = cp.sum(y == t) / 10000
        # print(accuracy)

        output = softmax(cp.dot(x_train, W_lin) + b_lin)
        y = cp.argmax(output, axis=1)
        t = cp.argmax(t_train, axis=1)
        train_accuracy = cp.sum(y == t) / 50000
        print(int(i / 100), accuracy, train_accuracy)

train_size = x_train.shape[0]
batch_size = 100
iter_per_epoch = 100
print("Linear classifier probe:hidden_layer1")
weight_init_std = 0.03
W_lin = weight_init_std * cp.random.randn(1000, 10)
b_lin = cp.zeros(10)
alpha = 0.01
for i in range(100000):
    batch_mask = cp.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]
    h = cp.dot(x_batch, BP_W1) + BP_b1
    h = cp.tanh(h)
    output = softmax(cp.dot(h, W_lin) + b_lin)
    delta = (output - t_batch) / batch_size
    delta_W_lin = cp.dot(h.T, delta)
    delta_b_lin = cp.dot(cp.ones(batch_size), delta)
    W_lin -= alpha * delta_W_lin
    b_lin -= alpha * delta_b_lin
    if i % 100 == 0:
        h = cp.dot(x_test, BP_W1) + BP_b1
        h = cp.tanh(h)
        output = softmax(cp.dot(h, W_lin) + b_lin)
        y = cp.argmax(output, axis=1)
        t = cp.argmax(t_test, axis=1)
        accuracy = cp.sum(y == t) / 10000

        h = cp.dot(x_train, BP_W1) + BP_b1
        h = cp.tanh(h)
        output = softmax(cp.dot(h, W_lin) + b_lin)
        y = cp.argmax(output, axis=1)
        t = cp.argmax(t_train, axis=1)
        train_accuracy = cp.sum(y == t) / 50000
        print(int(i / 100), accuracy, train_accuracy)

train_size = x_train.shape[0]
batch_size = 100
# iter_per_epoch = 100
print("Linear classifier probe:hidden_layer2")
weight_init_std = 0.03
W_lin = weight_init_std * cp.random.randn(1000, 10)
b_lin = cp.zeros(10)
alpha = 0.01
for i in range(100000):
    batch_mask = cp.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]
    h = cp.dot(x_batch, BP_W1) + BP_b1
    h = cp.tanh(h)
    h = cp.dot(h, BP_W2) + BP_b2
    h = cp.tanh(h)
    output = softmax(cp.dot(h, W_lin) + b_lin)
    delta = (output - t_batch) / batch_size
    delta_W_lin = cp.dot(h.T, delta)
    delta_b_lin = cp.dot(cp.ones(batch_size), delta)
    W_lin -= alpha * delta_W_lin
    b_lin -= alpha * delta_b_lin
    if i % 100 == 0:
        h = cp.dot(x_test, BP_W1) + BP_b1
        h = cp.tanh(h)
        h = cp.dot(h, BP_W2) + BP_b2
        h = cp.tanh(h)
        output = softmax(cp.dot(h, W_lin) + b_lin)
        y = cp.argmax(output, axis=1)
        t = cp.argmax(t_test, axis=1)
        accuracy = cp.sum(y == t) / 10000

        h = cp.dot(x_train, BP_W1) + BP_b1
        h = cp.tanh(h)
        h = cp.dot(h, BP_W2) + BP_b2
        h = cp.tanh(h)
        output = softmax(cp.dot(h, W_lin) + b_lin)
        y = cp.argmax(output, axis=1)
        t = cp.argmax(t_train, axis=1)
        train_accuracy = cp.sum(y == t) / 50000
        print(int(i / 100), accuracy, train_accuracy)

train_size = x_train.shape[0]
batch_size = 100
# iter_per_epoch = 100
print("Linear classifier probe:hidden_layer3")
weight_init_std = 0.03
W_lin = weight_init_std * cp.random.randn(1000, 10)
b_lin = cp.zeros(10)
alpha = 0.01
for i in range(100000):
    batch_mask = cp.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]
    h = cp.dot(x_batch, BP_W1) + BP_b1
    h = cp.tanh(h)
    h = cp.dot(h, BP_W2) + BP_b2
    h = cp.tanh(h)
    h = cp.dot(h, BP_W3) + BP_b3
    h = cp.tanh(h)
    output = softmax(cp.dot(h, W_lin) + b_lin)
    delta = (output - t_batch) / batch_size
    delta_W_lin = cp.dot(h.T, delta)
    delta_b_lin = cp.dot(cp.ones(batch_size), delta)
    W_lin -= alpha * delta_W_lin
    b_lin -= alpha * delta_b_lin
    if i % 100 == 0:
        h = cp.dot(x_test, BP_W1) + BP_b1
        h = cp.tanh(h)
        h = cp.dot(h, BP_W2) + BP_b2
        h = cp.tanh(h)
        h = cp.dot(h, BP_W3) + BP_b3
        h = cp.tanh(h)
        output = softmax(cp.dot(h, W_lin) + b_lin)
        y = cp.argmax(output, axis=1)
        t = cp.argmax(t_test, axis=1)
        accuracy = cp.sum(y == t) / 10000

        h = cp.dot(x_train, BP_W1) + BP_b1
        h = cp.tanh(h)
        h = cp.dot(h, BP_W2) + BP_b2
        h = cp.tanh(h)
        h = cp.dot(h, BP_W3) + BP_b3
        h = cp.tanh(h)
        output = softmax(cp.dot(h, W_lin) + b_lin)
        y = cp.argmax(output, axis=1)
        t = cp.argmax(t_train, axis=1)
        train_accuracy = cp.sum(y == t) / 50000
        print(int(i / 100), accuracy, train_accuracy)

train_size = x_train.shape[0]
batch_size = 100
# iter_per_epoch = 100
print("Linear classifier probe:hidden_layer4")
weight_init_std = 0.03
W_lin = weight_init_std * cp.random.randn(1000, 10)
b_lin = cp.zeros(10)
alpha = 0.01
for i in range(100000):
    batch_mask = cp.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]
    h = cp.dot(x_batch, BP_W1) + BP_b1
    h = cp.tanh(h)
    h = cp.dot(h, BP_W2) + BP_b2
    h = cp.tanh(h)
    h = cp.dot(h, BP_W3) + BP_b3
    h = cp.tanh(h)
    h = cp.dot(h, BP_W4) + BP_b4
    h = cp.tanh(h)
    output = softmax(cp.dot(h, W_lin) + b_lin)
    delta = (output - t_batch) / batch_size
    delta_W_lin = cp.dot(h.T, delta)
    delta_b_lin = cp.dot(cp.ones(batch_size), delta)
    W_lin -= alpha * delta_W_lin
    b_lin -= alpha * delta_b_lin
    if i % 100 == 0:
        h = cp.dot(x_test, BP_W1) + BP_b1
        h = cp.tanh(h)
        h = cp.dot(h, BP_W2) + BP_b2
        h = cp.tanh(h)
        h = cp.dot(h, BP_W3) + BP_b3
        h = cp.tanh(h)
        h = cp.dot(h, BP_W4) + BP_b4
        h = cp.tanh(h)
        output = softmax(cp.dot(h, W_lin) + b_lin)
        y = cp.argmax(output, axis=1)
        t = cp.argmax(t_test, axis=1)
        accuracy = cp.sum(y == t) / 10000

        h = cp.dot(x_train, BP_W1) + BP_b1
        h = cp.tanh(h)
        h = cp.dot(h, BP_W2) + BP_b2
        h = cp.tanh(h)
        h = cp.dot(h, BP_W3) + BP_b3
        h = cp.tanh(h)
        h = cp.dot(h, BP_W4) + BP_b4
        h = cp.tanh(h)
        output = softmax(cp.dot(h, W_lin) + b_lin)
        y = cp.argmax(output, axis=1)
        t = cp.argmax(t_train, axis=1)
        train_accuracy = cp.sum(y == t) / 50000
        print(int(i / 100), accuracy, train_accuracy)
