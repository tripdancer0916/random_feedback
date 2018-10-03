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
import argparse

cp.random.seed(100)

# Load the fashion-MNIST dataset
train, test = chainer.datasets.get_fashion_mnist()
x_train, t_train = train._datasets
x_test, t_test = test._datasets

x_train = cp.asarray(x_train)
x_test = cp.asarray(x_test)

t_train = cp.identity(10)[t_train.astype(int)]
t_test = cp.identity(10)[t_test.astype(int)]


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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Linear Classifier Probe')
    parser.add_argument('--use_epoch', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--iter_per_epoch', type=int, default=1000)

    args = parser.parse_args()

    dfa_W1 = cp.load("./fashion_model/dfa_{}_W_f1.npy".format(args.use_epoch))
    dfa_W2 = cp.load("./fashion_model/dfa_{}_W_f2.npy".format(args.use_epoch))
    dfa_W3 = cp.load("./fashion_model/dfa_{}_W_f3.npy".format(args.use_epoch))
    dfa_W4 = cp.load("./fashion_model/dfa_{}_W_f4.npy".format(args.use_epoch))
    dfa_W5 = cp.load("./fashion_model/dfa_{}_W_f5.npy".format(args.use_epoch))

    train_size = x_train.shape[0]
    batch_size = args.batch_size
    iter_per_epoch = args.iter_per_epoch
    hidden_unit = 800
    weight_init_std = 0.032
    alpha = 0.01

    print("Linear classifier probe: input")
    W_lin = weight_init_std * cp.random.randn(784, 10)
    b_lin = cp.zeros(10)
    for i in range(100000):
        batch_mask = cp.random.choice(train_size, batch_size)
        x_batch = x_train[batch_mask]
        t_batch = t_train[batch_mask]
        output = softmax(cp.dot(x_batch, W_lin) + b_lin)
        delta = (output - t_batch) / batch_size
        delta_W_lin = cp.dot(x_batch.T, delta)
        delta_b_lin = cp.dot(cp.ones(batch_size), delta)
        W_lin -= alpha * delta_W_lin
        b_lin -= alpha * delta_b_lin
        if i % iter_per_epoch == 0:
            output = softmax(cp.dot(x_test, W_lin) + b_lin)
            y = cp.argmax(output, axis=1)
            t = cp.argmax(t_test, axis=1)
            accuracy = cp.sum(y == t) / 10000

            output = softmax(cp.dot(x_train, W_lin) + b_lin)
            y = cp.argmax(output, axis=1)
            t = cp.argmax(t_train, axis=1)
            train_accuracy = cp.sum(y == t) / 60000
            print(int(i / iter_per_epoch), accuracy, train_accuracy)
    output = softmax(cp.dot(x_test, W_lin) + b_lin)
    y = cp.argmax(output, axis=1)
    t = cp.argmax(t_test, axis=1)
    accuracy = cp.sum(y == t) / 10000

    output = softmax(cp.dot(x_train, W_lin) + b_lin)
    y = cp.argmax(output, axis=1)
    t = cp.argmax(t_train, axis=1)
    train_accuracy = cp.sum(y == t) / 60000
    print(accuracy, train_accuracy)

    print("Linear classifier probe:hidden_layer1")
    W_lin = weight_init_std * cp.random.randn(hidden_unit, 10)
    b_lin = cp.zeros(10)
    for i in range(100000):
        batch_mask = cp.random.choice(train_size, batch_size)
        x_batch = x_train[batch_mask]
        t_batch = t_train[batch_mask]
        h = cp.dot(x_batch, dfa_W1)
        h = cp.tanh(h)
        output = softmax(cp.dot(h, W_lin) + b_lin)
        delta = (output - t_batch) / batch_size
        delta_W_lin = cp.dot(h.T, delta)
        delta_b_lin = cp.dot(cp.ones(batch_size), delta)
        W_lin -= alpha * delta_W_lin
        b_lin -= alpha * delta_b_lin
        if i % iter_per_epoch == 0:
            h = cp.dot(x_test, dfa_W1)
            h = cp.tanh(h)
            output = softmax(cp.dot(h, W_lin) + b_lin)
            y = cp.argmax(output, axis=1)
            t = cp.argmax(t_test, axis=1)
            accuracy = cp.sum(y == t) / 10000

            h = cp.dot(x_train, dfa_W1)
            h = cp.tanh(h)
            output = softmax(cp.dot(h, W_lin) + b_lin)
            y = cp.argmax(output, axis=1)
            t = cp.argmax(t_train, axis=1)
            train_accuracy = cp.sum(y == t) / 60000
            print(int(i / iter_per_epoch), accuracy, train_accuracy)
    output = softmax(cp.dot(x_test, W_lin) + b_lin)
    y = cp.argmax(output, axis=1)
    t = cp.argmax(t_test, axis=1)
    accuracy = cp.sum(y == t) / 10000

    output = softmax(cp.dot(x_train, W_lin) + b_lin)
    y = cp.argmax(output, axis=1)
    t = cp.argmax(t_train, axis=1)
    train_accuracy = cp.sum(y == t) / 60000
    print(accuracy, train_accuracy)

    print("Linear classifier probe:hidden_layer2")
    W_lin = weight_init_std * cp.random.randn(hidden_unit, 10)
    b_lin = cp.zeros(10)
    for i in range(100000):
        batch_mask = cp.random.choice(train_size, batch_size)
        x_batch = x_train[batch_mask]
        t_batch = t_train[batch_mask]
        h = cp.dot(x_batch, dfa_W1)
        h = cp.tanh(h)
        h = cp.dot(h, dfa_W2)
        h = cp.tanh(h)
        output = softmax(cp.dot(h, W_lin) + b_lin)
        delta = (output - t_batch) / batch_size
        delta_W_lin = cp.dot(h.T, delta)
        delta_b_lin = cp.dot(cp.ones(batch_size), delta)
        W_lin -= alpha * delta_W_lin
        b_lin -= alpha * delta_b_lin
        if i % iter_per_epoch == 0:
            h = cp.dot(x_test, dfa_W1)
            h = cp.tanh(h)
            h = cp.dot(h, dfa_W2)
            h = cp.tanh(h)
            output = softmax(cp.dot(h, W_lin) + b_lin)
            y = cp.argmax(output, axis=1)
            t = cp.argmax(t_test, axis=1)
            accuracy = cp.sum(y == t) / 10000

            h = cp.dot(x_train, dfa_W1)
            h = cp.tanh(h)
            h = cp.dot(h, dfa_W2)
            h = cp.tanh(h)
            output = softmax(cp.dot(h, W_lin) + b_lin)
            y = cp.argmax(output, axis=1)
            t = cp.argmax(t_train, axis=1)
            train_accuracy = cp.sum(y == t) / 60000
            print(int(i / iter_per_epoch), accuracy, train_accuracy)
    output = softmax(cp.dot(x_test, W_lin) + b_lin)
    y = cp.argmax(output, axis=1)
    t = cp.argmax(t_test, axis=1)
    accuracy = cp.sum(y == t) / 10000

    output = softmax(cp.dot(x_train, W_lin) + b_lin)
    y = cp.argmax(output, axis=1)
    t = cp.argmax(t_train, axis=1)
    train_accuracy = cp.sum(y == t) / 60000
    print(accuracy, train_accuracy)

    print("Linear classifier probe:hidden_layer3")
    W_lin = weight_init_std * cp.random.randn(hidden_unit, 10)
    b_lin = cp.zeros(10)
    for i in range(100000):
        batch_mask = cp.random.choice(train_size, batch_size)
        x_batch = x_train[batch_mask]
        t_batch = t_train[batch_mask]
        h = cp.dot(x_batch, dfa_W1)
        h = cp.tanh(h)
        h = cp.dot(h, dfa_W2)
        h = cp.tanh(h)
        h = cp.dot(h, dfa_W3)
        h = cp.tanh(h)
        output = softmax(cp.dot(h, W_lin) + b_lin)
        delta = (output - t_batch) / batch_size
        delta_W_lin = cp.dot(h.T, delta)
        delta_b_lin = cp.dot(cp.ones(batch_size), delta)
        W_lin -= alpha * delta_W_lin
        b_lin -= alpha * delta_b_lin
        if i % iter_per_epoch == 0:
            h = cp.dot(x_test, dfa_W1)
            h = cp.tanh(h)
            h = cp.dot(h, dfa_W2)
            h = cp.tanh(h)
            h = cp.dot(h, dfa_W3)
            h = cp.tanh(h)
            output = softmax(cp.dot(h, W_lin) + b_lin)
            y = cp.argmax(output, axis=1)
            t = cp.argmax(t_test, axis=1)
            accuracy = cp.sum(y == t) / 10000

            h = cp.dot(x_train, dfa_W1)
            h = cp.tanh(h)
            h = cp.dot(h, dfa_W2)
            h = cp.tanh(h)
            h = cp.dot(h, dfa_W3)
            h = cp.tanh(h)
            output = softmax(cp.dot(h, W_lin) + b_lin)
            y = cp.argmax(output, axis=1)
            t = cp.argmax(t_train, axis=1)
            train_accuracy = cp.sum(y == t) / 60000
            print(int(i / iter_per_epoch), accuracy, train_accuracy)
    output = softmax(cp.dot(x_test, W_lin) + b_lin)
    y = cp.argmax(output, axis=1)
    t = cp.argmax(t_test, axis=1)
    accuracy = cp.sum(y == t) / 10000

    output = softmax(cp.dot(x_train, W_lin) + b_lin)
    y = cp.argmax(output, axis=1)
    t = cp.argmax(t_train, axis=1)
    train_accuracy = cp.sum(y == t) / 60000
    print(accuracy, train_accuracy)

    print("Linear classifier probe:hidden_layer4")
    W_lin = weight_init_std * cp.random.randn(hidden_unit, 10)
    b_lin = cp.zeros(10)
    alpha = 0.01
    for i in range(100000):
        batch_mask = cp.random.choice(train_size, batch_size)
        x_batch = x_train[batch_mask]
        t_batch = t_train[batch_mask]
        h = cp.dot(x_batch, dfa_W1)
        h = cp.tanh(h)
        h = cp.dot(h, dfa_W2)
        h = cp.tanh(h)
        h = cp.dot(h, dfa_W3)
        h = cp.tanh(h)
        h = cp.dot(h, dfa_W4)
        h = cp.tanh(h)
        output = softmax(cp.dot(h, W_lin) + b_lin)
        delta = (output - t_batch) / batch_size
        delta_W_lin = cp.dot(h.T, delta)
        delta_b_lin = cp.dot(cp.ones(batch_size), delta)
        W_lin -= alpha * delta_W_lin
        b_lin -= alpha * delta_b_lin
        if i % iter_per_epoch == 0:
            h = cp.dot(x_test, dfa_W1)
            h = cp.tanh(h)
            h = cp.dot(h, dfa_W2)
            h = cp.tanh(h)
            h = cp.dot(h, dfa_W3)
            h = cp.tanh(h)
            h = cp.dot(h, dfa_W4)
            h = cp.tanh(h)
            output = softmax(cp.dot(h, W_lin) + b_lin)
            y = cp.argmax(output, axis=1)
            t = cp.argmax(t_test, axis=1)
            accuracy = cp.sum(y == t) / 10000

            h = cp.dot(x_train, dfa_W1)
            h = cp.tanh(h)
            h = cp.dot(h, dfa_W2)
            h = cp.tanh(h)
            h = cp.dot(h, dfa_W3)
            h = cp.tanh(h)
            h = cp.dot(h, dfa_W4)
            h = cp.tanh(h)
            output = softmax(cp.dot(h, W_lin) + b_lin)
            y = cp.argmax(output, axis=1)
            t = cp.argmax(t_train, axis=1)
            train_accuracy = cp.sum(y == t) / 60000
            print(int(i / iter_per_epoch), accuracy, train_accuracy)
    output = softmax(cp.dot(x_test, W_lin) + b_lin)
    y = cp.argmax(output, axis=1)
    t = cp.argmax(t_test, axis=1)
    accuracy = cp.sum(y == t) / 10000

    output = softmax(cp.dot(x_train, W_lin) + b_lin)
    y = cp.argmax(output, axis=1)
    t = cp.argmax(t_train, axis=1)
    train_accuracy = cp.sum(y == t) / 60000
    print(accuracy, train_accuracy)

