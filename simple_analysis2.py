# coding:utf-8

import numpy as np
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
import os
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
hidden_unit = 2000


class MLP:
    def __init__(self, weight_init_std=0.01):
        self.W_f1 = weight_init_std * cp.random.randn(784, hidden_unit)
        self.W_f2 = weight_init_std * cp.random.randn(hidden_unit, 10)
        # self.B1 = weight_init_std * cp.random.randn(10, hidden_unit)
        # self.B2 = weight_init_std * cp.random.randn(hidden_unit, 784)
        variable = [-1, 1]
        self.d = np.random.choice(variable, 10)
        self.B = []
        for i in range(hidden_unit):
            coordinate = np.random.choice(variable)
            self.B.append(coordinate * self.d)
        self.B = weight_init_std * cp.array(self.B)
        self.B = self.B.T

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

        delta2 = (output - target) / batch_size
        delta_Wf2 = cp.dot(h1_.T, delta2)

        delta1 = relu_grad(h1) * cp.dot(delta2, self.W_f2.T)

        delta_Wf1 = cp.dot(x.T, delta1)

        alpha = 0.1
        self.W_f1 -= alpha * delta_Wf1
        self.W_f2 -= alpha * delta_Wf2

    def feedback_alignment(self, x, target):
        h1 = cp.dot(x, self.W_f1)
        h1_ = relu(h1)
        h2 = cp.dot(h1_, self.W_f2)
        output = softmax(h2)

        delta2 = (output - target) / batch_size
        delta_Wf2 = cp.dot(h1_.T, delta2)

        delta1 = relu_grad(h1) * cp.dot(delta2, self.B)
        delta_Wf1 = cp.dot(x.T, delta1)

        alpha = 0.1
        self.W_f1 -= alpha * delta_Wf1
        self.W_f2 -= alpha * delta_Wf2


mlp = MLP()
train_loss_list_FA = []
test_loss_list_FA = []
train_acc_list_FA = []
test_acc_list_FA = []
output = []

train_size = x_train.shape[0]
batch_size = 100
iter_per_epoch = 100
for i in range(100):
    batch_mask = cp.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]
    # mlp.gradient(x_batch, t_batch)
    mlp.feedback_alignment(x_batch,t_batch)
    train_acc = mlp.accuracy(x_train, t_train)
    test_acc = mlp.accuracy(x_test, t_test)
    train_loss = mlp.loss(x_train, t_train)
    test_loss = mlp.loss(x_test, t_test)
    train_loss_list_FA.append(cuda.to_cpu(train_loss))
    test_loss_list_FA.append(cuda.to_cpu(test_loss))
    train_acc_list_FA.append(cuda.to_cpu(train_acc))
    test_acc_list_FA.append(cuda.to_cpu(test_acc))
    # print("epoch:", int(i / iter_per_epoch), " train acc, test acc | " + str(train_acc) + ", " + str(test_acc))
    output.append(cuda.to_cpu(mlp.predict(x_train[0])))
    # global_error_variable = np.dot(cuda.to_cpu(mlp.d), (cuda.to_cpu(mlp.predict(x_batch[:10]))-cuda.to_cpu(t_batch[:10])).T)
    # compared_error_variable = np.dot(np.ones(10), (cuda.to_cpu(mlp.predict(x_batch[:10]))-cuda.to_cpu(t_batch[:10])).T)
    # print(np.mean(global_error_variable), np.mean(compared_error_variable))
    # print(np.var(global_error_variable), np.var(compared_error_variable))


plt.plot(output[0], label="iter:0")
plt.plot(output[10], label="iter:10")
plt.plot(output[30], label="iter:30")
plt.plot(output[99], label="iter:99")
plt.title("transition of output probability distribution")
plt.legend()
os.makedirs("./result/0717", exist_ok=True)
plt.savefig("./result/0717/output_probability_distribution")


"""
plt.plot(train_acc_list_FA, label="RFA train acc", linestyle="dotted", color="orange")
plt.plot(test_acc_list_FA, label="RFA test acc", color="orange")
plt.title("BP/RFA for MNIST")
plt.legend()

plt.savefig("./result/BP-RFA_for_mnist.png")
plt.figure()
plt.plot(train_acc_list[20:], label="BP train acc", linestyle="dotted", color="blue")
plt.plot(test_acc_list[20:], label="BP test acc", color="blue")
# plt.title("BP for MNIST")
# plt.legend()

# plt.savefig("mnistBP.png")


plt.plot(train_acc_list_FA[20:], label="RFA train acc", linestyle="dashed", color="orange")
plt.plot(test_acc_list_FA[20:], label="RFA test acc", color="orange")
plt.title("BP/RFA for MNIST relu")
plt.legend()

plt.savefig("./result/BP-RFA_for_mnist_20start.png")
"""