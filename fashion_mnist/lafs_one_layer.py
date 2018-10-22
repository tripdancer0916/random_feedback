# coding:utf-8

import numpy as np
import os
import keras
from keras import backend as K
from keras.datasets import cifar10
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
import matplotlib as mpl
import argparse

cp.random.seed(100)

mpl.use('Agg')
import matplotlib.pyplot as plt

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


# Network definition


class LAFS:
    def __init__(self, weight_init_std=0.032, hidden_unit=800):
        self.h = [0, 0]

        self.W_f1 = cp.zeros([784, hidden_unit])
        self.W_f2 = cp.zeros([hidden_unit, hidden_unit])

        self.dB = weight_init_std * cp.random.randn(10, hidden_unit)

    def predict(self, x):
        self.h[0] = cp.dot(x, self.W_f1)
        self.h[0] = cp.tanh(self.h[0])
        h = cp.dot(self.h[0], self.W_f2)
        output = softmax(h)
        return output

    def accuracy(self, x, t):
        y = self.predict(x)
        y = cp.argmax(y, axis=1)
        t = cp.argmax(t, axis=1)
        accuracy = cp.sum(y == t) / float(x.shape[0])
        return accuracy

    def hidden_acc(self, x, i, t):
        self.predict(x)
        y = cp.dot(self.h[i], self.dB[i].T)
        y = softmax(y)
        y = cp.argmax(y, axis=1)
        t = cp.argmax(t, axis=1)
        accuracy = cp.sum(y == t) / float(x.shape[0])
        return accuracy

    def loss(self, x, t):
        y = self.predict(x)
        return cross_entropy_error(y, t)

    def lafs(self, x, target, batch_size):
        h1 = cp.dot(x, self.W_f1)
        h1_ = cp.tanh(h1)
        output1 = softmax(cp.dot(h1_, self.dB[0].T))

        h = cp.dot(h1_, self.W_f2)
        output = softmax(h)

        delta = (output - target) / batch_size
        delta_Wf2 = cp.dot(h1_.T, delta)

        delta1 = tanh_grad(h1) * cp.dot((output1 - target) / batch_size, self.dB[0])
        delta_Wf1 = cp.dot(x.T, delta1)

        alpha = 0.1
        self.W_f1 -= alpha * delta_Wf1
        self.W_f2 -= alpha * delta_Wf2


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Direct Feedback Alignment.')
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--used_data', type=int, default=60000)
    parser.add_argument('--iter_per_epoch', type=int, default=500)
    # parser.add_argument('--model_save', type=bool, default=False)
    parser.add_argument('--n_unit', type=int, default=200)

    args = parser.parse_args()

    mlp = LAFS(weight_init_std=0.032, hidden_unit=args.n_unit)
    train_loss_list = []
    test_loss_list = []
    train_acc_list = []
    test_acc_list = []

    train_size = x_train.shape[0]
    batch_size = args.batch_size

    iter_per_epoch = args.iter_per_epoch
    batch_mask_ = cp.random.choice(train_size, batch_size, replace=False)
    x_batch = x_train[batch_mask_]
    t_batch = t_train[batch_mask_]
    mlp.lafs(x_batch, t_batch, batch_size)
    hidden_train_acc = [[float(mlp.hidden_acc(x_train, j, t_train))] for j in range(4)]
    train_acc_list.append(float(mlp.accuracy(x_train, t_train)))
    for i in range(1000000):
        batch_mask_ = cp.random.choice(args.used_data, batch_size, replace=False)
        x_batch = x_train[batch_mask_]
        t_batch = t_train[batch_mask_]
        mlp.lafs(x_batch, t_batch, batch_size)
        if i % iter_per_epoch == 0:
            train_acc = mlp.accuracy(x_train, t_train)
            train_acc_list.append(float(train_acc))
            test_acc = mlp.accuracy(x_test, t_test)