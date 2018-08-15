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
import keras
from keras.datasets import cifar10
import matplotlib as mpl

mpl.use('Agg')
import matplotlib.pyplot as plt

os.makedirs('./result/0815/', exist_ok=True)

# Load the MNIST dataset
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


# Network definition
hidden_unit1 = 4000
hidden_unit2 = 3000
hidden_unit3 = 2000
hidden_unit4 = 1000


class MLP:
    def __init__(self, weight_init_std=0.01):
        self.W_f1 = weight_init_std * cp.random.randn(3072, hidden_unit1)
        self.W_f2 = weight_init_std * cp.random.randn(hidden_unit1, hidden_unit2)
        self.W_f3 = weight_init_std * cp.random.randn(hidden_unit2, hidden_unit3)
        self.W_f4 = weight_init_std * cp.random.randn(hidden_unit3, hidden_unit4)
        self.W_f5 = weight_init_std * cp.random.randn(hidden_unit4, 10)

        # self.W_f1 = weight_init_std * cp.zeros([3072, hidden_unit1])
        # self.W_f2 = weight_init_std * cp.zeros([hidden_unit1, hidden_unit2])
        # self.W_f3 = weight_init_std * cp.zeros([hidden_unit2, hidden_unit3])
        # self.W_f4 = weight_init_std * cp.zeros([hidden_unit3, hidden_unit4])
        # self.W_f5 = weight_init_std * cp.zeros([hidden_unit4, 10])

        self.b1 = weight_init_std * cp.random.randn(hidden_unit1)
        self.b2 = weight_init_std * cp.random.randn(hidden_unit2)
        self.b3 = weight_init_std * cp.random.randn(hidden_unit3)
        self.b4 = weight_init_std * cp.random.randn(hidden_unit4)
        self.b5 = weight_init_std * cp.random.randn(10)

        self.B2 = weight_init_std * cp.random.randn(hidden_unit2, hidden_unit1)
        self.B3 = weight_init_std * cp.random.randn(hidden_unit3, hidden_unit2)
        self.B4 = weight_init_std * cp.random.randn(hidden_unit4, hidden_unit3)
        self.B5 = weight_init_std * cp.random.randn(10, hidden_unit4)

        # self.B2 = cp.random.normal(0, 0.01, hidden_unit1 * hidden_unit2).reshape(hidden_unit2, hidden_unit1)
        # self.B3 = cp.random.normal(0, 0.01, hidden_unit2 * hidden_unit3).reshape(hidden_unit3, hidden_unit2)
        # self.B4 = cp.random.normal(0, 0.01, hidden_unit3 * hidden_unit4).reshape(hidden_unit4, hidden_unit3)
        # self.B5 = cp.random.normal(0, 0.01, hidden_unit4 * 10).reshape(10, hidden_unit4)

        self.h_W1 = 0
        self.h_W2 = 0
        self.h_W3 = 0
        self.h_W4 = 0
        self.h_W5 = 0
        self.h_b1 = 0
        self.h_b2 = 0
        self.h_b3 = 0
        self.h_b4 = 0
        self.h_b5 = 0

    def predict(self, x):
        h1 = cp.dot(x, self.W_f1) + self.b1
        h1 = cp.tanh(h1)
        h2 = cp.dot(h1, self.W_f2) + self.b2
        h2 = cp.tanh(h2)
        h3 = cp.dot(h2, self.W_f3) + self.b3
        h3 = cp.tanh(h3)
        h4 = cp.dot(h3, self.W_f4) + self.b4
        h4 = cp.tanh(h4)
        h5 = cp.dot(h4, self.W_f5) + self.b5
        output = softmax(h5)
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
        h1 = cp.dot(x, self.W_f1) + self.b1
        h1_ = cp.tanh(h1)
        h2 = cp.dot(h1_, self.W_f2) + self.b2
        h2_ = cp.tanh(h2)
        h3 = cp.dot(h2_, self.W_f3) + self.b3
        h3_ = cp.tanh(h3)
        h4 = cp.dot(h3_, self.W_f4) + self.b4
        h4_ = cp.tanh(h4)
        h5 = cp.dot(h4_, self.W_f5) + self.b5
        output = softmax(h5)

        delta5 = (output - target) / batch_size
        self.delta_Wf5 = cp.dot(h4_.T, delta5)
        self.delta_b5 = cp.dot(cp.ones(batch_size), delta5)

        delta4 = tanh_grad(h4) * cp.dot(delta5, self.W_f5.T)
        self.delta_Wf4 = cp.dot(h3_.T, delta4)
        self.delta_b4 = cp.dot(cp.ones(batch_size), delta4)

        delta3 = tanh_grad(h3) * cp.dot(delta4, self.W_f4.T)
        self.delta_Wf3 = cp.dot(h2_.T, delta3)
        self.delta_b3 = cp.dot(cp.ones(batch_size), delta3)

        delta2 = tanh_grad(h2) * cp.dot(delta3, self.W_f3.T)
        self.delta_Wf2 = cp.dot(h1_.T, delta2)
        self.delta_b2 = cp.dot(cp.ones(batch_size), delta2)

        delta1 = tanh_grad(h1) * cp.dot(delta2, self.W_f2.T)
        self.delta_Wf1 = cp.dot(x.T, delta1)
        self.delta_b1 = cp.dot(cp.ones(batch_size), delta1)
        # print(delta_Wf1)
        eta, self.h_W1 = self.rms_prop(self.delta_Wf1, self.h_W1)
        self.W_f1 -= eta * self.delta_Wf1
        eta, self.h_W2 = self.rms_prop(self.delta_Wf2, self.h_W2)
        self.W_f2 -= eta * self.delta_Wf2
        eta, self.h_W3 = self.rms_prop(self.delta_Wf3, self.h_W3)
        self.W_f3 -= eta * self.delta_Wf3
        eta, self.h_W4 = self.rms_prop(self.delta_Wf4, self.h_W4)
        self.W_f4 -= eta * self.delta_Wf4
        eta, self.h_W5 = self.rms_prop(self.delta_Wf5, self.h_W5)
        self.W_f5 -= eta * self.delta_Wf5
        eta, self.h_b1 = self.rms_prop(self.delta_b1, self.h_b1)
        self.b1 -= eta * self.delta_b1
        eta, self.h_b2 = self.rms_prop(self.delta_b2, self.h_b2)
        self.b2 -= eta * self.delta_b2
        eta, self.h_b3 = self.rms_prop(self.delta_b3, self.h_b3)
        self.b3 -= eta * self.delta_b3
        eta, self.h_b4 = self.rms_prop(self.delta_b4, self.h_b4)
        self.b4 -= eta * self.delta_b4
        eta, self.h_b5 = self.rms_prop(self.delta_b5, self.h_b5)
        self.b5 -= eta * self.delta_b5

    def rms_prop(self, grad, h):
        alpha = 0.99
        eta_0 = 0.02
        epsilon = 1e-8
        quad_grad = cp.linalg.norm(grad)**2
        h = alpha * h + (1-alpha)*quad_grad
        eta = eta_0 / (cp.sqrt(h) + epsilon)
        return eta, h

    def learning_rate(self, epoch):
        if epoch <= 20000:
            return 0.12
        elif epoch <= 30000:
            return 0.1
        else:
            return 0.08

    def feedback_alignment(self, x, target, epoch):
        h1 = cp.dot(x, self.W_f1) + self.b1
        h1_ = cp.tanh(h1)
        h2 = cp.dot(h1_, self.W_f2) + self.b2
        h2_ = cp.tanh(h2)
        h3 = cp.dot(h2_, self.W_f3) + self.b3
        h3_ = cp.tanh(h3)
        h4 = cp.dot(h3_, self.W_f4) + self.b4
        h4_ = cp.tanh(h4)
        h5 = cp.dot(h4_, self.W_f5) + self.b5
        output = softmax(h5)

        delta5 = (output - target) / batch_size
        delta_Wf5 = cp.dot(h4_.T, delta5)
        delta_b5 = cp.dot(cp.ones(batch_size), delta5)

        delta4 = tanh_grad(h4) * cp.dot(delta5, self.B5)
        delta_Wf4 = cp.dot(h3_.T, delta4)
        delta_b4 = cp.dot(cp.ones(batch_size), delta4)

        delta3 = tanh_grad(h3) * cp.dot(delta4, self.B4)
        delta_Wf3 = cp.dot(h2_.T, delta3)
        delta_b3 = cp.dot(cp.ones(batch_size), delta3)

        delta2 = tanh_grad(h2) * cp.dot(delta3, self.B3)
        delta_Wf2 = cp.dot(h1_.T, delta2)
        delta_b2 = cp.dot(cp.ones(batch_size), delta2)

        delta1 = tanh_grad(h1) * cp.dot(delta2, self.B2)
        delta_Wf1 = cp.dot(x.T, delta1)
        delta_b1 = cp.dot(cp.ones(batch_size), delta1)
        # print(delta_Wf1)

        alpha1 = self.learning_rate(epoch)
        self.W_f1 -= alpha1 * delta_Wf1
        self.W_f2 -= alpha1 * delta_Wf2
        self.W_f3 -= alpha1 * delta_Wf3
        self.W_f4 -= alpha1 * delta_Wf4
        self.W_f5 -= alpha1 * delta_Wf5
        self.b1 -= alpha1 * delta_b1
        self.b2 -= alpha1 * delta_b2
        self.b3 -= alpha1 * delta_b3
        self.b4 -= alpha1 * delta_b4
        self.b5 -= alpha1 * delta_b5


mlp = MLP()
train_loss_list = []
test_loss_list = []
train_acc_list = []
test_acc_list = []

train_size = x_train.shape[0]
batch_size = 100
iter_per_epoch = 100
print("Back propagation")
for i in range(50000):
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

np.savetxt("./result/0815/BP_cifarW1.txt", cuda.to_cpu(mlp.W_f1))
np.savetxt("./result/0815/BP_cifarW2.txt", cuda.to_cpu(mlp.W_f2))
np.savetxt("./result/0815/BP_cifarW3.txt", cuda.to_cpu(mlp.W_f3))
np.savetxt("./result/0815/BP_cifarW4.txt", cuda.to_cpu(mlp.W_f4))
np.savetxt("./result/0815/BP_cifarW5.txt", cuda.to_cpu(mlp.W_f5))
"""

mlp = MLP()
train_loss_list_FA = []
test_loss_list_FA = []
train_acc_list_FA = []
test_acc_list_FA = []

train_size = x_train.shape[0]
batch_size = 100
iter_per_epoch = 100
print("Feedback alignment")
for i in range(200000):
    batch_mask = cp.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]
    # mlp.gradient(x_batch, t_batch)
    mlp.feedback_alignment(x_batch, t_batch, i)

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

np.savetxt("./result/0815/FA_cifarW1.txt", cuda.to_cpu(mlp.W_f1))
np.savetxt("./result/0815/FA_cifarW2.txt", cuda.to_cpu(mlp.W_f2))
np.savetxt("./result/0815/FA_cifarW3.txt", cuda.to_cpu(mlp.W_f3))
np.savetxt("./result/0815/FA_cifarW4.txt", cuda.to_cpu(mlp.W_f4))
np.savetxt("./result/0815/FA_cifarW5.txt", cuda.to_cpu(mlp.W_f5))
"""