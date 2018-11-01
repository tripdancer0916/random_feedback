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


class MLP:
    def __init__(self, weight_init_std=0.032, hidden_unit=200):
        self.h = [0, 0, 0, 0]

        self.W_f1 = weight_init_std * cp.random.randn(784, hidden_unit)
        self.W_f2 = weight_init_std * cp.random.randn(hidden_unit, hidden_unit)
        self.W_f3 = weight_init_std * cp.random.randn(hidden_unit, hidden_unit)
        self.W_f4 = weight_init_std * cp.random.randn(hidden_unit, hidden_unit)
        self.W_f5 = weight_init_std * cp.random.randn(hidden_unit, 10)

        self.dB = weight_init_std * cp.random.randn(4, 10, hidden_unit)

    def predict(self, x):
        self.h[0] = cp.dot(x, self.W_f1)
        self.h[0] = relu(self.h[0])
        self.h[1] = cp.dot(self.h[0], self.W_f2)
        self.h[1] = relu(self.h[1])
        self.h[2] = cp.dot(self.h[1], self.W_f3)
        self.h[2] = relu(self.h[2])
        self.h[3] = cp.dot(self.h[2], self.W_f4)
        self.h[3] = relu(self.h[3])
        h = cp.dot(self.h[3], self.W_f5)
        output = softmax(h)
        return output

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

    def direct_feedback_alignment(self, x, target, batch_size, alpha):
        h1 = cp.dot(x, self.W_f1)
        h1_ = relu(h1)
        h2 = cp.dot(h1_, self.W_f2)
        h2_ = relu(h2)
        h3 = cp.dot(h2_, self.W_f3)
        h3_ = relu(h3)
        h4 = cp.dot(h3_, self.W_f4)
        h4_ = relu(h4)
        h5 = cp.dot(h4_, self.W_f5)
        output = softmax(h5)

        delta5 = (output - target) / batch_size
        delta_Wf5 = cp.dot(h4_.T, delta5)

        delta4 = relu_grad(h4) * cp.dot(delta5, self.dB[3])
        delta_Wf4 = cp.dot(h3_.T, delta4)
        delta3 = relu_grad(h3) * cp.dot(delta5, self.dB[2])
        delta_Wf3 = cp.dot(h2_.T, delta3)
        delta2 = relu_grad(h2) * cp.dot(delta5, self.dB[1])
        delta_Wf2 = cp.dot(h1_.T, delta2)
        delta1 = relu_grad(h1) * cp.dot(delta5, self.dB[0])
        delta_Wf1 = cp.dot(x.T, delta1)

        self.W_f1 -= alpha * delta_Wf1
        self.W_f2 -= alpha * delta_Wf2
        self.W_f3 -= alpha * delta_Wf3
        self.W_f4 -= alpha * delta_Wf4
        self.W_f5 -= alpha * delta_Wf5


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Direct Feedback Alignment.')
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--n_unit', type=int, default=800)
    parser.add_argument('--learning_rate', type=float, default=0.02)

    args = parser.parse_args()

    mlp = MLP(weight_init_std=0.032, hidden_unit=args.n_unit)
    train_loss_list = []
    test_loss_list = []
    train_acc_list = []
    test_acc_list = []

    train_size = x_train.shape[0]
    batch_size = args.batch_size

    iter_per_epoch = 500
    print("measure accuracy of hidden-layer in the dynamics of DFA learning.")
    batch_mask_ = cp.random.choice(train_size, batch_size, replace=False)
    x_batch = x_train[batch_mask_]
    t_batch = t_train[batch_mask_]
    mlp.direct_feedback_alignment(x_batch, t_batch, batch_size, args.learning_rate)
    hidden_train_acc = [[float(mlp.hidden_acc(x_train, j, t_train))] for j in range(4)]
    train_acc_list.append(float(mlp.accuracy(x_train, t_train)))
    for i in range(500000):
        batch_mask_ = cp.random.choice(train_size, batch_size, replace=False)
        x_batch = x_train[batch_mask_]
        t_batch = t_train[batch_mask_]
        mlp.direct_feedback_alignment(x_batch, t_batch, batch_size, args.learning_rate)
        if i % iter_per_epoch == 0:
            train_acc = mlp.accuracy(x_train, t_train)
            linear_train_acc = mlp.linear_accuracy(x_train, t_train)
            linear_test_acc = mlp.linear_accuracy(x_test, t_test)
            train_acc_list.append(float(train_acc))
            test_acc = mlp.accuracy(x_test, t_test)
            for j in range(4):
                hidden_train_acc[j].append(float(mlp.hidden_acc(x_train, j, t_train)))
            print(int(i / iter_per_epoch), 'train_acc: ', train_acc, 'test_acc: ', test_acc)
            print(int(i / iter_per_epoch), 'linear_train_acc: ', linear_train_acc, 'linear_test_acc: ', linear_test_acc)
            # print('hidden_train_acc_1: ', hidden_train_acc[0][int(i / iter_per_epoch)+1])
            # print('hidden_train_acc_2: ', hidden_train_acc[1][int(i / iter_per_epoch)+1])
            # print('hidden_train_acc_3: ', hidden_train_acc[2][int(i / iter_per_epoch)+1])
            # print('hidden_train_acc_4: ', hidden_train_acc[3][int(i / iter_per_epoch)+1])

    cp.save('./weights_dfa_relu_W_f1', mlp.W_f1)
    cp.save('./weights_dfa_relu_W_f2', mlp.W_f2)
    cp.save('./weights_dfa_relu_W_f3', mlp.W_f3)
    cp.save('./weights_dfa_relu_W_f4', mlp.W_f4)
    cp.save('./weights_dfa_relu_W_f5', mlp.W_f5)

