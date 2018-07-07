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
import os
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
        self.W_f3 = weight_init_std * cp.random.randn(hidden_unit, 10)
        self.B3 = weight_init_std * cp.random.randn(10, hidden_unit)
        self.B2 = weight_init_std * cp.random.randn(hidden_unit, hidden_unit)
        self.delta_Wf1 = None
        self.delta_Wf2 = None

        self.delta2_bp = None
        self.delta1_bp = None
        self.delta_Wf2bp = None
        self.delta_Wf1bp = None

    def predict(self, x):
        h1 = cp.dot(x, self.W_f1)
        h1 = relu(h1)
        h2 = cp.dot(h1, self.W_f2)
        h2 = relu(h2)
        h3 = cp.dot(h2, self.W_f3)
        output = softmax(h3)
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
        output = softmax(h3)

        delta3 = (output - target) / batch_size
        delta_Wf3 = cp.dot(h2_.T, delta3)

        delta2 = relu_grad(h2) * cp.dot(delta3, self.W_f3.T)

        delta_Wf2 = cp.dot(h1_.T, delta2)

        delta1 = relu_grad(h1) * cp.dot(delta2, self.W_f2.T)

        delta_Wf1 = cp.dot(x.T, delta1)

        alpha = 0.1
        self.W_f1 -= alpha * delta_Wf1
        self.W_f2 -= alpha * delta_Wf2
        self.W_f3 -= alpha * delta_Wf3

    def feedback_alignment(self, x, target):
        h1 = cp.dot(x, self.W_f1)
        h1_ = relu(h1)
        h2 = cp.dot(h1_, self.W_f2)
        h2_ = relu(h2)
        h3 = cp.dot(h2_, self.W_f3)
        output = softmax(h3)

        delta3 = (output - target) / batch_size
        delta_Wf3 = cp.dot(h2_.T, delta3)
        delta2 = relu_grad(h2) * cp.dot(delta3, self.B3)
        self.delta_Wf2 = cp.dot(h1_.T, delta2)
        delta1 = relu_grad(h1) * cp.dot(delta2, self.B2)
        self.delta_Wf1 = cp.dot(x.T, delta1)

        self.delta2_bp = relu_grad(h2) * cp.dot(delta3, self.W_f3.T)
        self.delta_Wf2bp = cp.dot(h1_.T, self.delta2_bp)
        self.delta1_bp = relu_grad(h1) * cp.dot(self.delta2_bp, self.W_f2.T)
        self.delta_Wf1bp = cp.dot(x.T, self.delta1_bp)

        alpha = 0.1
        self.W_f1 -= alpha * self.delta_Wf1
        self.W_f2 -= alpha * self.delta_Wf2
        self.W_f3 -= alpha * delta_Wf3

    def angle(self, A, B):
        fp = A * B
        fp = cp.sum(fp)
        norm_a = cp.sqrt(cp.sum(A * A))
        norm_b = cp.sqrt(cp.sum(B * B))
        cos_theta = fp / (norm_a * norm_b)
        return cp.arccos(cos_theta)


mlp = MLP()
# train_loss_list_FA = []
# test_loss_list_FA = []
# train_acc_list_FA = []
# test_acc_list_FA = []

angle_W2 = []
angle_W3 = []
angle_dW1 = []
angle_dW2 = []

print("FA")
train_size = x_train.shape[0]
batch_size = 100
iter_per_epoch = 100
for i in range(50000):
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
        angle_W2.append(cuda.to_cpu(mlp.angle(mlp.W_f2.T, mlp.B2)))
        print(mlp.angle(mlp.W_f2.T, mlp.B2))
        angle_W3.append(cuda.to_cpu(mlp.angle(mlp.W_f3.T, mlp.B3)))
        angle_dW1.append(cuda.to_cpu(mlp.angle(mlp.delta_Wf1bp, mlp.delta_Wf1)))
        angle_dW2.append(cuda.to_cpu(mlp.angle(mlp.delta_Wf2bp, mlp.delta_Wf2)))
        print("epoch:", int(i / iter_per_epoch), " train acc, test acc | " + str(train_acc) + ", " + str(test_acc))


# plt.plot(test_acc_list_FA, label="RFA", color="orange")

# plt.plot(train_acc_list_l, label="only last layer", linestyle="dashed", color="orange")
# plt.plot(test_acc_list_l, label="only last layer", color="green")


plt.figure()
plt.plot(angle_W2)
plt.plot([0,500], [np.pi/2, np.pi/2], "black", linestyles='dashed')
plt.title(r"angle between $W^{(2)T}$ and $B^{(2)}$")
plt.xlabel("epoch")
plt.ylabel("angle")
plt.set_ylim([0, 3.14])
plt.legend()

os.makedirs('./result/0707/', exist_ok=True)
plt.savefig("./result/0707/angle_W2.png")

plt.figure()
plt.plot(angle_W3)
plt.plot([0,500], [np.pi/2, np.pi/2], "black", linestyles='dashed')
plt.title(r"angle between $W^{(3)T}$ and $B^{(3)}$")
plt.xlabel("epoch")
plt.ylabel("angle")
plt.set_ylim([0, 3.14])
plt.legend()

plt.savefig("./result/0707/angle_W3.png")

plt.figure()
plt.plot(angle_dW1)
plt.plot([0,500], [np.pi/2, np.pi/2], "black", linestyles='dashed')
plt.title(r"angle between $dW_{FA}^{(1)}$ and $dW_{BP}^{(1)}$")
plt.xlabel("epoch")
plt.ylabel("angle")
plt.set_ylim([0, 3.14])
plt.legend()

plt.savefig("./result/0707/angle_dW1.png")

plt.figure()
plt.plot(angle_dW2)
plt.plot([0,500], [np.pi/2, np.pi/2], "black", linestyles='dashed')
plt.title(r"angle between $dW_{FA}^{(2)}$ and $dW_{BP}^{(2)}$")
plt.xlabel("epoch")
plt.ylabel("angle")
plt.set_ylim([0, 3.14])
plt.legend()

plt.savefig("./result/0707/angle_dW2.png")
