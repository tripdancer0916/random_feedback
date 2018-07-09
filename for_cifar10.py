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

mpl.use('Agg')
import matplotlib.pyplot as plt

num_classes = 10

(x_train, t_train), (x_test, t_test) = cifar10.load_data()

# x_train = x_train.reshape(())
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

x_train = x_train.reshape(-1, 3072)
x_test = x_test.reshape(-1, 3072)

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
        self.W_f1 = weight_init_std * cp.random.randn(3072, hidden_unit)
        self.W_f2 = weight_init_std * cp.random.randn(hidden_unit, hidden_unit)
        self.W_f3 = weight_init_std * cp.random.randn(hidden_unit, hidden_unit)
        self.W_f4 = weight_init_std * cp.random.randn(hidden_unit, 10)

        self.allones = weight_init_std * cp.ones([10, hidden_unit])

        self.d1 = np.random.rand(10) * 2 - 1
        self.d2 = np.random.rand(10) * 2 - 1
        self.d3 = np.random.rand(10) * 2 - 1

        self.B3 = weight_init_std * cp.random.randn(10, hidden_unit)
        self.B2 = weight_init_std * cp.random.randn(10, hidden_unit)
        self.B1 = weight_init_std * cp.random.randn(10, hidden_unit)

        self.B3_ge1 = []
        for i in range(1000):
            magnification = np.random.rand() * 2 - 1
            self.B3_ge1.append(magnification * self.d1)
        self.B3_ge1 = weight_init_std * cp.array(self.B3_ge1)
        self.B3_ge1 = self.B3_ge1.T

        self.B2_ge1 = []
        for i in range(1000):
            magnification = np.random.rand() * 2 - 1
            self.B2_ge1.append(magnification * self.d1)
        self.B2_ge1 = weight_init_std * cp.array(self.B2_ge1)
        self.B2_ge1 = self.B2_ge1.T

        self.B1_ge1 = []
        for i in range(1000):
            magnification = np.random.rand() * 2 - 1
            self.B1_ge1.append(magnification * self.d1)
        self.B1_ge1 = weight_init_std * cp.array(self.B1_ge1)
        self.B1_ge1 = self.B1_ge1.T

        tmp2 = [0, 1]
        self.B3_ge2 = []
        for i in range(hidden_unit):
            magnification = np.random.rand() * 2 - 1
            selecter = np.random.choice(tmp2)
            if selecter == 0:
                self.B3_ge2.append(magnification * self.d1)
            else:
                self.B3_ge2.append(magnification * self.d2)
        self.B3_ge2 = weight_init_std * cp.array(self.B3_ge2)
        self.B3_ge2 = self.B3_ge2.T
        self.B2_ge2 = []
        for i in range(hidden_unit):
            magnification = np.random.rand() * 2 - 1
            selecter = np.random.choice(tmp2)
            if selecter == 0:
                self.B2_ge2.append(magnification * self.d1)
            else:
                self.B2_ge2.append(magnification * self.d2)
        self.B2_ge2 = weight_init_std * cp.array(self.B2_ge2)
        self.B2_ge2 = self.B2_ge2.T
        self.B1_ge2 = []
        for i in range(hidden_unit):
            magnification = np.random.rand() * 2 - 1
            selecter = np.random.choice(tmp2)
            if selecter == 0:
                self.B1_ge2.append(magnification * self.d1)
            else:
                self.B1_ge2.append(magnification * self.d2)
        self.B1_ge2 = weight_init_std * cp.array(self.B1_ge2)
        self.B1_ge2 = self.B1_ge2.T

        tmp3 = [0, 1, 2]
        self.B3_ge3 = []
        for i in range(hidden_unit):
            magnification = np.random.rand() * 2 - 1
            selecter = np.random.choice(tmp3)
            if selecter == 0:
                self.B3_ge3.append(magnification * self.d1)
            elif selecter == 1:
                self.B3_ge3.append(magnification * self.d2)
            else:
                self.B3_ge3.append(magnification * self.d3)
        self.B3_ge3 = weight_init_std * cp.array(self.B3_ge3)
        self.B3_ge3 = self.B3_ge3.T
        self.B2_ge3 = []
        for i in range(hidden_unit):
            magnification = np.random.rand() * 2 - 1
            selecter = np.random.choice(tmp3)
            if selecter == 0:
                self.B2_ge3.append(magnification * self.d1)
            elif selecter == 1:
                self.B2_ge3.append(magnification * self.d2)
            else:
                self.B2_ge3.append(magnification * self.d3)
        self.B2_ge3 = weight_init_std * cp.array(self.B2_ge3)
        self.B2_ge3 = self.B2_ge3.T
        self.B1_ge3 = []
        for i in range(hidden_unit):
            magnification = np.random.rand() * 2 - 1
            selecter = np.random.choice(tmp3)
            if selecter == 0:
                self.B1_ge3.append(magnification * self.d1)
            elif selecter == 1:
                self.B1_ge3.append(magnification * self.d2)
            else:
                self.B1_ge3.append(magnification * self.d3)
        self.B1_ge3 = weight_init_std * cp.array(self.B1_ge3)
        self.B1_ge3 = self.B1_ge3.T

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

    def direct_FA(self, x, target):
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
        delta_Wf3 = cp.dot(h2_.T, relu_grad(h3) * delta3)

        delta2 = cp.dot(delta4, self.B2)
        delta_Wf2 = cp.dot(h1_.T, relu_grad(h2) * delta2)

        delta1 = cp.dot(delta4, self.B1)
        delta_Wf1 = cp.dot(x.T, relu_grad(h1) * delta1)

        alpha1 = 0.1
        self.W_f1 -= alpha1 * delta_Wf1
        self.W_f2 -= alpha1 * delta_Wf2
        self.W_f3 -= alpha1 * delta_Wf3
        self.W_f4 -= alpha1 * delta_Wf4

    def global_error1(self, x, target):
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

        delta3 = cp.dot(delta4, self.B3_ge1)
        delta_Wf3 = cp.dot(h2_.T, relu_grad(h3) * delta3)

        delta2 = cp.dot(delta4, self.B2_ge1)
        delta_Wf2 = cp.dot(h1_.T, relu_grad(h2) * delta2)

        delta1 = cp.dot(delta4, self.B1_ge1)
        delta_Wf1 = cp.dot(x.T, relu_grad(h1) * delta1)

        alpha1 = 0.1
        self.W_f1 -= alpha1 * delta_Wf1
        self.W_f2 -= alpha1 * delta_Wf2
        self.W_f3 -= alpha1 * delta_Wf3
        self.W_f4 -= alpha1 * delta_Wf4

    def global_error2(self, x, target):
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

        delta3 = cp.dot(delta4, self.B3_ge2)
        delta_Wf3 = cp.dot(h2_.T, relu_grad(h3) * delta3)

        delta2 = cp.dot(delta4, self.B2_ge2)
        delta_Wf2 = cp.dot(h1_.T, relu_grad(h2) * delta2)

        delta1 = cp.dot(delta4, self.B1_ge2)
        delta_Wf1 = cp.dot(x.T, relu_grad(h1) * delta1)

        alpha1 = 0.1
        self.W_f1 -= alpha1 * delta_Wf1
        self.W_f2 -= alpha1 * delta_Wf2
        self.W_f3 -= alpha1 * delta_Wf3
        self.W_f4 -= alpha1 * delta_Wf4

    def global_error3(self, x, target):
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

        delta3 = cp.dot(delta4, self.B3_ge3)
        delta_Wf3 = cp.dot(h2_.T, relu_grad(h3) * delta3)

        delta2 = cp.dot(delta4, self.B2_ge3)
        delta_Wf2 = cp.dot(h1_.T, relu_grad(h2) * delta2)

        delta1 = cp.dot(delta4, self.B1_ge3)
        delta_Wf1 = cp.dot(x.T, relu_grad(h1) * delta1)

        alpha1 = 0.1
        self.W_f1 -= alpha1 * delta_Wf1
        self.W_f2 -= alpha1 * delta_Wf2
        self.W_f3 -= alpha1 * delta_Wf3
        self.W_f4 -= alpha1 * delta_Wf4

"""
mlp = MLP()
test_acc_list_bp = []
print("backpropagation")
train_size = x_train.shape[0]
batch_size = 100
iter_per_epoch = 100
for i in range(100000):
    batch_mask = cp.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]
    mlp.gradient(x_batch, t_batch)
    if i % iter_per_epoch == 0:
        train_acc = mlp.accuracy(x_train, t_train)
        test_acc = mlp.accuracy(x_test, t_test)
        train_loss = mlp.loss(x_train, t_train)
        test_loss = mlp.loss(x_test, t_test)
        test_acc_list_bp.append(cuda.to_cpu(test_acc))
        print("epoch:", int(i / iter_per_epoch), " train loss, test loss, train acc, test acc | " + str(train_loss)
              + ", " + str(test_loss) + ", " + str(train_acc) + ", " + str(test_acc))
"""
mlp = MLP()
test_acc_list_DFA = []
print("direct feedback alignment")
train_size = x_train.shape[0]
batch_size = 100
iter_per_epoch = 100
for i in range(100000):
    batch_mask = cp.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]
    mlp.direct_FA(x_batch, t_batch)
    if i % iter_per_epoch == 0:
        train_acc = mlp.accuracy(x_train, t_train)
        test_acc = mlp.accuracy(x_test, t_test)
        train_loss = mlp.loss(x_train, t_train)
        test_loss = mlp.loss(x_test, t_test)
        test_acc_list_DFA.append(cuda.to_cpu(test_acc))
        print("epoch:", int(i / iter_per_epoch), " train loss, test loss, train acc, test acc | " + str(train_loss)
              + ", " + str(test_loss) + ", " + str(train_acc) + ", " + str(test_acc))

mlp = MLP()
test_acc_list_ge1 = []
print("global error 1")
train_size = x_train.shape[0]
batch_size = 100
iter_per_epoch = 100
for i in range(100000):
    batch_mask = cp.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]
    mlp.global_error1(x_batch, t_batch)
    if i % iter_per_epoch == 0:
        train_acc = mlp.accuracy(x_train, t_train)
        test_acc = mlp.accuracy(x_test, t_test)
        train_loss = mlp.loss(x_train, t_train)
        test_loss = mlp.loss(x_test, t_test)
        test_acc_list_ge1.append(cuda.to_cpu(test_acc))
        print("epoch:", int(i / iter_per_epoch), " train loss, test loss, train acc, test acc | " + str(train_loss)
              + ", " + str(test_loss) + ", " + str(train_acc) + ", " + str(test_acc))

mlp = MLP()
test_acc_list_ge2 = []
print("global error 2")
train_size = x_train.shape[0]
batch_size = 100
iter_per_epoch = 100
for i in range(100000):
    batch_mask = cp.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]
    mlp.global_error2(x_batch, t_batch)
    if i % iter_per_epoch == 0:
        train_acc = mlp.accuracy(x_train, t_train)
        test_acc = mlp.accuracy(x_test, t_test)
        train_loss = mlp.loss(x_train, t_train)
        test_loss = mlp.loss(x_test, t_test)
        test_acc_list_ge2.append(cuda.to_cpu(test_acc))
        print("epoch:", int(i / iter_per_epoch), " train loss, test loss, train acc, test acc | " + str(train_loss)
              + ", " + str(test_loss) + ", " + str(train_acc) + ", " + str(test_acc))

mlp = MLP()
test_acc_list_ge3 = []
print("global error 3")
train_size = x_train.shape[0]
batch_size = 100
iter_per_epoch = 100
for i in range(100000):
    batch_mask = cp.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]
    mlp.global_error3(x_batch, t_batch)
    if i % iter_per_epoch == 0:
        train_acc = mlp.accuracy(x_train, t_train)
        test_acc = mlp.accuracy(x_test, t_test)
        train_loss = mlp.loss(x_train, t_train)
        test_loss = mlp.loss(x_test, t_test)
        test_acc_list_ge3.append(cuda.to_cpu(test_acc))
        print("epoch:", int(i / iter_per_epoch), " train loss, test loss, train acc, test acc | " + str(train_loss)
              + ", " + str(test_loss) + ", " + str(train_acc) + ", " + str(test_acc))


plt.figure()
plt.plot(test_acc_list_ge1, label="K=1", color="crimson")
plt.plot(test_acc_list_ge2, label="K=2", color="darkblue")
plt.plot(test_acc_list_ge3, label="K=3", color="green")
plt.plot(test_acc_list_bp, label="backprop", color="plum")
plt.plot(test_acc_list_DFA, label="K=1000", color="grey")


plt.title("test accuracy for MNIST")
plt.xlabel("epoch")
plt.ylabel("acc")
plt.legend()

os.makedirs('./result/0709/', exist_ok=True)
plt.savefig("./result/0709/cifar10.png")
