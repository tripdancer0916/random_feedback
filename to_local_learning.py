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
mpl.use('Agg')
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
        self.W_f2 = weight_init_std * cp.random.randn(hidden_unit, hidden_unit)
        self.W_f3 = weight_init_std * cp.random.randn(hidden_unit, hidden_unit)
        self.W_f4 = weight_init_std * cp.random.randn(hidden_unit, 10)
        """
        self.B3 = cp.random.randn(10, hidden_unit)
        self.B3[self.B3 > 0] = 1
        self.B3[self.B3 < 0] = -1
        self.B3 = weight_init_std * self.B3
        """
        # tmp = [-1, 1]
        # d = np.random.choice(tmp, 10)
        d = np.random.rand(10) * 2 - 1
        # d *= weight_init_std
        self.B3 = []
        for i in range(1000):
            magnification = np.random.rand() * 2 - 1
            self.B3.append(d*magnification)
        self.B3 = weight_init_std * cp.array(self.B3)
        self.B3 = self.B3.T

        self.B2 = []
        for i in range(1000):
            magnification = np.random.rand() * 2 - 1
            self.B2.append(d * magnification)
        self.B2 = weight_init_std * cp.array(self.B2)
        self.B2 = self.B2.T

        self.B1 = []
        for i in range(1000):
            magnification = np.random.rand() * 2 - 1
            self.B1.append(d * magnification)
        self.B1 = weight_init_std * cp.array(self.B1)
        self.B1 = self.B1.T
        # self.B3 = weight_init_std * cp.ones([10, hidden_unit])
        # for i in range(10):
        #     if cp.random.rand() > 0.5:
        #         self.B3[i] *= -1

        # self.B3 = self.B3.T
        # print(self.B3)
        # self.B2 = cp.random.randn(10, hidden_unit)
        # self.B2[self.B2 > 0] = 1
        # self.B2[self.B2 < 0] = -1
        # self.B2 = weight_init_std * self.B2

        # self.B3 = weight_init_std * cp.ones([10, hidden_unit])
        # self.B2 = weight_init_std * cp.ones([10, hidden_unit])

    def predict(self, x):
        h1 = cp.dot(x, self.W_f1)
        h1 = relu(h1)
        h2 = cp.dot(h1, self.W_f2)
        h2 = relu(h2)
        h3 = cp.dot(h2, self.W_f3)
        h3 = relu(h3)
        h4 = cp.dot(h3, self.W_f4)
        output = softmax(h4)
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
        h2_ = relu(h2)
        h3 = cp.dot(h2_, self.W_f3)
        h3_ = relu(h3)
        h4 = cp.dot(h3_, self.W_f4)
        output = softmax(h4)

        delta4 = (output - target) / batch_size
        delta_Wf4 = cp.dot(h3_.T, delta4)

        delta3 = cp.dot(delta4, self.W_f4.T)
        delta_Wf3 = cp.dot(h2_.T, delta3)

        delta2 = cp.dot(delta3, self.W_f3.T)
        delta_Wf2 = cp.dot(h1_.T, relu_grad(h2) * delta2)

        delta1 = cp.dot(delta2, self.W_f2.T)
        delta_Wf1 = cp.dot(x.T, relu_grad(h1) * delta1)

        alpha = 0.1
        self.W_f1 -= alpha * delta_Wf1
        self.W_f2 -= alpha * delta_Wf2
        self.W_f3 -= alpha * delta_Wf3
        self.W_f4 -= alpha * delta_Wf4

    def feedback_alignment(self, x, target):
        h1 = cp.dot(x, self.W_f1)
        h1_ = relu(h1)
        h2 = cp.dot(h1_, self.W_f2)
        h2_ = relu(h2)
        h3 = cp.dot(h2_, self.W_f3)
        h3_ = relu(h3)
        h4 = cp.dot(h3_, self.W_f4)
        output = softmax(h4)

        delta4 = (output - target) / batch_size
        delta_Wf4 = cp.dot(h3_.T, delta4)

        delta3 = cp.dot(delta4, self.B3)
        delta_Wf3 = cp.dot(h2_.T, delta3)

        delta2 = cp.dot(delta4, self.B2)
        delta_Wf2 = cp.dot(h1_.T, relu_grad(h2) * delta2)

        delta1 = cp.dot(delta4, self.B1)
        delta_Wf1 = cp.dot(x.T, relu_grad(h1) * delta1)

        alpha1 = 0.1
        # alpha2 = 0.1
        # alpha3 = 0.05
        # alpha4 = 0.03
        self.W_f1 -= alpha1 * delta_Wf1
        self.W_f2 -= alpha1 * delta_Wf2
        self.W_f3 -= alpha1 * delta_Wf3
        self.W_f4 -= alpha1 * delta_Wf4

"""
mlp = MLP()
train_loss_list = []
test_loss_list = []
train_acc_list = []
test_acc_list = []

train_size = x_train.shape[0]
batch_size = 100
iter_per_epoch = 100
for i in range(20000):
    batch_mask = cp.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]
    mlp.gradient(x_batch, t_batch)
    # mlp.feedback_alignment(x_batch,t_batch)

    if i % iter_per_epoch == 0:
        train_acc = mlp.accuracy(x_train, t_train)
        test_acc = mlp.accuracy(x_test, t_test)
        train_loss = mlp.loss(x_train, t_train)
        test_loss = mlp.loss(x_test, t_test)
        train_loss_list.append(cuda.to_cpu(train_loss))
        test_loss_list.append(cuda.to_cpu(test_loss))
        train_acc_list.append(cuda.to_cpu(train_acc))
        test_acc_list.append(cuda.to_cpu(test_acc))
        print("epoch:", int(i / iter_per_epoch), " train acc, test acc | " + str(train_acc) + ", " + str(test_acc))
"""

mlp = MLP()
train_loss_list_FA = []
test_loss_list_FA = []
train_acc_list_FA = []
test_acc_list_FA = []

train_size = x_train.shape[0]
batch_size = 100
iter_per_epoch = 100
for i in range(100000):
    batch_mask = cp.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]
    # mlp.gradient(x_batch, t_batch)
    mlp.feedback_alignment(x_batch, t_batch)

    if i % iter_per_epoch == 0:
        train_acc = mlp.accuracy(x_train, t_train)
        test_acc = mlp.accuracy(x_test, t_test)
        train_loss = mlp.loss(x_train, t_train)
        test_loss = mlp.loss(x_test, t_test)
        train_loss_list_FA.append(cuda.to_cpu(train_loss))
        test_loss_list_FA.append(cuda.to_cpu(test_loss))
        train_acc_list_FA.append(cuda.to_cpu(train_acc))
        test_acc_list_FA.append(cuda.to_cpu(test_acc))
        print("epoch:", int(i / iter_per_epoch), " train acc, test acc | " + str(train_acc) + ", " + str(test_acc))
"""
plt.plot(train_acc_list, label="BP train acc", linestyle="dashed", color="blue")
plt.plot(test_acc_list, label="BP test acc", color="blue")
# plt.title("BP for MNIST")
# plt.legend()

# plt.savefig("mnistBP.png")

plt.plot(train_acc_list_FA, label="DFA train acc", linestyle="dotted", color="orange")
plt.plot(test_acc_list_FA, label="DFA test acc", color="orange")
plt.title("BP/DFA for MNIST relu")
plt.legend()

plt.savefig("./result/BP-DFA_for_mnist.png")
"""
plt.figure()
# plt.plot(train_acc_list[20:], label="BP train acc", linestyle="dotted", color="blue")
# plt.plot(test_acc_list[20:], label="BP test acc", color="blue")
# plt.title("BP for MNIST")
# plt.legend()

# plt.savefig("mnistBP.png")


plt.plot(train_acc_list_FA[20:], label="DFA train acc", linestyle="dotted", color="orange")
plt.plot(test_acc_list_FA[20:], label="DFA test acc", color="orange")
plt.title("DFA for MNIST relu start from 20")
plt.legend()

os.makedirs('./result/0706/', exist_ok=True)

plt.savefig("./result/0706/to_local_learning5.png")
