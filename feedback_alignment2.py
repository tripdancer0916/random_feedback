# coding:utf-8

import numpy as cp
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
import matplotlib.pyplot as plt

# Load the MNIST dataset
train, test = chainer.datasets.get_mnist()
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


def softmax(x):
    if x.ndim == 2:
        x = x.T
        x = x - cp.max(x, axis=0)
        y = cp.exp(x) / cp.sum(cp.exp(x), axis=0)
        return y.T

    x = x - cp.max(x)
    return cp.exp(x) / cp.sum(cp.exp(x))


# Network definition
hidden_unit = 1000


class MLP:
    def __init__(self, weight_init_std=0.01):
        self.W_f1 = weight_init_std * cp.random.randn(784, hidden_unit)
        self.W_f2 = weight_init_std * cp.random.randn(hidden_unit, 10)
        self.B1 = weight_init_std * cp.random.randn(10, hidden_unit)
        self.B2 = weight_init_std * cp.random.randn(hidden_unit, 784)

    def predict(self, x):
        h1 = cp.dot(x, self.W_f1)
        h1 = relu(h1)
        h2 = cp.dot(h1, self.W_f2)
        output = softmax(h2)
        return output

    def accuracy(self, x, t):
        y = self.predict(x)
        y = cp.argmax(y, axis=1)
        t = cp.argmax(t, axis=1)

        accuracy = cp.sum(y == t) / float(x.shape[0])
        return accuracy

    def loss(self, x, t):
        y = self.predict(x)
        return cross_entropy_error(y, t)

    def gradient(self, x, target):
        h1 = cp.dot(x, self.W_f1)
        h1_ = relu(h1)
        h2 = cp.dot(h1_, self.W_f2)
        output = softmax(h2)

        delta2 = (output - target) / 100
        delta_Wf2 = cp.dot(h1_.T, delta2)

        delta1 = cp.dot(delta2, self.W_f2.T)

        delta_Wf1 = cp.dot(x.T, relu_grad(h1) * delta1)

        alpha = 0.1
        self.W_f1 -= alpha * delta_Wf1
        self.W_f2 -= alpha * delta_Wf2

    def feedback_alignment(self, x, target):
        h1 = cp.dot(x, self.W_f1)
        h1_ = relu(h1)
        h2 = cp.dot(h1_, self.W_f2)
        output = softmax(h2)

        delta2 = (output - target) / 100
        delta_Wf2 = cp.dot(h1_.T, delta2)

        delta1 = cp.dot(delta2, self.B1)

        delta_Wf1 = cp.dot(x.T, relu_grad(h1) * delta1)

        alpha = 0.1
        self.W_f1 -= alpha * delta_Wf1
        self.W_f2 -= alpha * delta_Wf2


mlp = MLP()
train_loss_list = []
test_loss_list = []
train_acc_list = []
test_acc_list = []

train_size = x_train.shape[0]
batch_size = 100
iter_per_epoch = 100
for i in range(10000):
    batch_mask = cp.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]
    # mlp.gradient(x_batch, t_batch)
    mlp.feedback_alignment(x_batch,t_batch)

    if i % iter_per_epoch == 0:
        train_acc = mlp.accuracy(x_train, t_train)
        test_acc = mlp.accuracy(x_test, t_test)
        train_loss = mlp.loss(x_train, t_train)
        test_loss = mlp.loss(x_test, t_test)
        train_loss_list.append(train_loss)
        test_loss_list.append(test_loss)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print("epoch:", int(i / iter_per_epoch), " train acc, test acc | " + str(train_acc) + ", " + str(test_acc))
