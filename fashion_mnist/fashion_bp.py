# coding:utf-8

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
hidden_unit = 800


class MLP:
    def __init__(self, weight_init_std=0.032):
        self.h = [0, 0, 0, 0]

        self.W_f1 = weight_init_std * cp.random.randn(784, hidden_unit)
        self.W_f2 = weight_init_std * cp.random.randn(hidden_unit, hidden_unit)
        self.W_f3 = weight_init_std * cp.random.randn(hidden_unit, hidden_unit)
        self.W_f4 = weight_init_std * cp.random.randn(hidden_unit, hidden_unit)
        self.W_f5 = weight_init_std * cp.random.randn(hidden_unit, 10)

        self.dB = weight_init_std * cp.random.randn(4, 10, hidden_unit)

    def predict(self, x):
        self.h[0] = cp.dot(x, self.W_f1)
        self.h[0] = cp.tanh(self.h[0])
        self.h[1] = cp.dot(self.h[0], self.W_f2)
        self.h[1] = cp.tanh(self.h[1])
        self.h[2] = cp.dot(self.h[1], self.W_f3)
        self.h[2] = cp.tanh(self.h[2])
        self.h[3] = cp.dot(self.h[2], self.W_f4)
        self.h[3] = cp.tanh(self.h[3])
        h = cp.dot(self.h[3], self.W_f5)
        output = softmax(h)
        return output

    def accuracy(self, x, t):
        y = self.predict(x)
        y = cp.argmax(y, axis=1)
        t = cp.argmax(t, axis=1)
        accuracy = cp.sum(y == t) / float(x.shape[0])
        return accuracy

    def linear_predict(self, x):
        self.h[0] = cp.dot(x, self.W_f1)
        self.h[1] = cp.dot(self.h[0], self.W_f2)
        self.h[2] = cp.dot(self.h[1], self.W_f3)
        self.h[3] = cp.dot(self.h[2], self.W_f4)
        h = cp.dot(self.h[3], self.W_f5)
        output = softmax(h)
        return output

    def linear_accuracy(self, x, t):
        y = self.linear_predict(x)
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

    def angle(self, a, b):
        A = cp.dot(a,b)
        B = cp.linalg.norm(a)
        C = cp.linalg.norm(b)
        t = A/(B*C)
        s = cp.arccos(t)
        return (s/np.pi)*180

    def loss(self, x, t):
        y = self.predict(x)
        return cross_entropy_error(y, t)

    def direct_feedback_alignment(self, x, target, batch_size):
        h1 = cp.dot(x, self.W_f1)
        h1_ = cp.tanh(h1)
        h2 = cp.dot(h1_, self.W_f2)
        h2_ = cp.tanh(h2)
        h3 = cp.dot(h2_, self.W_f3)
        h3_ = cp.tanh(h3)
        h4 = cp.dot(h3_, self.W_f4)
        h4_ = cp.tanh(h4)
        h5 = cp.dot(h4_, self.W_f5)
        output = softmax(h5)

        delta5 = (output - target) / batch_size
        delta_Wf5 = cp.dot(h4_.T, delta5)

        delta4 = tanh_grad(h4) * cp.dot(delta5, self.dB[3])
        delta_Wf4 = cp.dot(h3_.T, delta4)
        delta3 = tanh_grad(h3) * cp.dot(delta5, self.dB[2])
        delta_Wf3 = cp.dot(h2_.T, delta3)
        delta2 = tanh_grad(h2) * cp.dot(delta5, self.dB[1])
        delta_Wf2 = cp.dot(h1_.T, delta2)
        delta1 = tanh_grad(h1) * cp.dot(delta5, self.dB[0])
        delta_Wf1 = cp.dot(x.T, delta1)

        alpha = 0.1
        self.W_f1 -= alpha * delta_Wf1
        self.W_f2 -= alpha * delta_Wf2
        self.W_f3 -= alpha * delta_Wf3
        self.W_f4 -= alpha * delta_Wf4
        self.W_f5 -= alpha * delta_Wf5

    def back_propagation(self, x, target, batch_size, alpha):
        h1 = cp.dot(x, self.W_f1)
        h1_ = cp.tanh(h1)
        h2 = cp.dot(h1_, self.W_f2)
        h2_ = cp.tanh(h2)
        h3 = cp.dot(h2_, self.W_f3)
        h3_ = cp.tanh(h3)
        h4 = cp.dot(h3_, self.W_f4)
        h4_ = cp.tanh(h4)
        h5 = cp.dot(h4_, self.W_f5)
        output = softmax(h5)

        delta5 = (output - target) / batch_size
        delta_Wf5 = cp.dot(h4_.T, delta5)

        delta4 = tanh_grad(h4) * cp.dot(delta5, self.W_f5.T)
        delta_Wf4 = cp.dot(h3_.T, delta4)
        norm_Wf4 = cp.linalg.norm(delta_Wf4)
        delta_Wf4 += (norm_Wf4/540) * cp.random.randn(800,800)

        delta3 = tanh_grad(h3) * cp.dot(delta4, self.W_f4.T)
        delta_Wf3 = cp.dot(h2_.T, delta3)
        norm_Wf3 = cp.linalg.norm(delta_Wf3)
        delta_Wf3 += (norm_Wf3/540) * cp.random.randn(800,800)

        delta2 = tanh_grad(h2) * cp.dot(delta3, self.W_f3.T)
        delta_Wf2 = cp.dot(h1_.T, delta2)
        norm_Wf2 = cp.linalg.norm(delta_Wf2)
        delta_Wf2 += (norm_Wf2/540) * cp.random.randn(800,800)

        delta1 = tanh_grad(h1) * cp.dot(delta2, self.W_f2.T)
        delta_Wf1 = cp.dot(x.T, delta1)
        norm_Wf1 = cp.linalg.norm(delta_Wf1)
        delta_Wf1 += (norm_Wf1/540) * cp.random.randn(784,800)

        self.W_f1 -= alpha * delta_Wf1
        self.W_f2 -= alpha * delta_Wf2
        self.W_f3 -= alpha * delta_Wf3
        self.W_f4 -= alpha * delta_Wf4
        self.W_f5 -= alpha * delta_Wf5


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Direct Feedback Alignment.')
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--used_data', type=int, default=60000)
    parser.add_argument('--iter_per_epoch', type=int, default=500)
    parser.add_argument('--learning_rate', type=float, default=0.05)

    args = parser.parse_args()

    mlp = MLP()
    train_loss_list = []
    test_loss_list = []
    train_acc_list = []
    test_acc_list = []

    train_size = x_train.shape[0]
    batch_size = args.batch_size

    iter_per_epoch = args.iter_per_epoch
    print("the dynamics of Back propagation.")
    batch_mask = cp.random.choice(train_size, args.used_data, replace=False)
    x_batch_ = x_train[batch_mask]
    t_batch_ = t_train[batch_mask]
    batch_mask_ = cp.random.choice(args.used_data, batch_size, replace=False)
    x_batch_tmp = x_batch_[batch_mask_]
    t_batch_tmp = t_batch_[batch_mask_]
    mlp.back_propagation(x_batch_tmp, t_batch_tmp, batch_size, args.learning_rate)
    hidden_train_acc = [[float(mlp.hidden_acc(x_batch_, j, t_batch_))] for j in range(4)]
    train_acc_list.append(float(mlp.accuracy(x_train, t_train)))
    for i in range(100000):
        batch_mask_ = cp.random.choice(args.used_data, batch_size, replace=False)
        x_batch = x_batch_[batch_mask_]
        t_batch = t_batch_[batch_mask_]
        mlp.back_propagation(x_batch, t_batch, batch_size, args.learning_rate)
        if i % iter_per_epoch == 0:
            train_acc = mlp.accuracy(x_train, t_train)
            linear_train_acc = mlp.linear_accuracy(x_train, t_train)
            linear_test_acc = mlp.linear_accuracy(x_test, t_test)
            train_acc_list.append(float(train_acc))
            test_acc = mlp.accuracy(x_test, t_test)
            print(int(i / iter_per_epoch), 'train_acc: ', train_acc, 'test_acc: ', test_acc)
            # print(int(i / iter_per_epoch), 'linear_train_acc: ', linear_train_acc, 'linear_test_acc: ', linear_test_acc)
