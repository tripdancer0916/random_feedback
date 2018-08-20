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

os.makedirs('./result/0820/', exist_ok=True)

# Load the MNIST dataset
num_classes = 10

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
hidden_unit1 = 1000
hidden_unit2 = 1000
hidden_unit3 = 1000
hidden_unit4 = 1000


class MLP:
    def __init__(self, weight_init_std=0.032):
        self.W_f1 = weight_init_std * cp.random.randn(784, hidden_unit1)
        self.W_f2 = weight_init_std * cp.random.randn(hidden_unit1, hidden_unit2)
        self.W_f3 = weight_init_std * cp.random.randn(hidden_unit2, hidden_unit3)
        self.W_f4 = weight_init_std * cp.random.randn(hidden_unit3, hidden_unit4)
        self.W_f5 = weight_init_std * cp.random.randn(hidden_unit4, hidden_unit1)
        self.W_f6 = weight_init_std * cp.random.randn(hidden_unit1, hidden_unit1)
        self.W_f7 = weight_init_std * cp.random.randn(hidden_unit1, hidden_unit1)
        self.W_f8 = weight_init_std * cp.random.randn(hidden_unit1, hidden_unit1)
        self.W_f9 = weight_init_std * cp.random.randn(hidden_unit1, hidden_unit1)
        self.W_f10 = weight_init_std * cp.random.randn(hidden_unit1, 10)

        # self.W_f1 = weight_init_std * cp.zeros([3072, hidden_unit1])
        # self.W_f2 = weight_init_std * cp.zeros([hidden_unit1, hidden_unit2])
        # self.W_f3 = weight_init_std * cp.zeros([hidden_unit2, hidden_unit3])
        # self.W_f4 = weight_init_std * cp.zeros([hidden_unit3, hidden_unit4])
        # self.W_f5 = weight_init_std * cp.zeros([hidden_unit4, 10])

        self.b1 = weight_init_std * cp.zeros(hidden_unit1)
        self.b2 = weight_init_std * cp.zeros(hidden_unit2)
        self.b3 = weight_init_std * cp.zeros(hidden_unit3)
        self.b4 = weight_init_std * cp.zeros(hidden_unit4)
        self.b5 = weight_init_std * cp.zeros(hidden_unit1)
        self.b6 = weight_init_std * cp.zeros(hidden_unit1)
        self.b7 = weight_init_std * cp.zeros(hidden_unit1)
        self.b8 = weight_init_std * cp.zeros(hidden_unit1)
        self.b9 = weight_init_std * cp.zeros(hidden_unit1)
        self.b10 = weight_init_std * cp.zeros(10)

        self.B2 = weight_init_std * cp.random.randn(hidden_unit2, hidden_unit1)
        self.B3 = weight_init_std * cp.random.randn(hidden_unit3, hidden_unit2)
        self.B4 = weight_init_std * cp.random.randn(hidden_unit4, hidden_unit3)
        self.B5 = weight_init_std * cp.random.randn(hidden_unit1, hidden_unit4)
        self.B6 = weight_init_std * cp.random.randn(hidden_unit1, hidden_unit4)
        self.B7 = weight_init_std * cp.random.randn(hidden_unit1, hidden_unit4)
        self.B8 = weight_init_std * cp.random.randn(hidden_unit1, hidden_unit4)
        self.B9 = weight_init_std * cp.random.randn(hidden_unit1, hidden_unit4)
        self.B10 = weight_init_std * cp.random.randn(10, hidden_unit4)

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
        h5 = cp.tanh(h5)
        h6 = cp.dot(h5, self.W_f6) + self.b6
        h6 = cp.tanh(h6)
        h7 = cp.dot(h6, self.W_f7) + self.b7
        h7 = cp.tanh(h7)
        h8 = cp.dot(h7, self.W_f8) + self.b8
        h8 = cp.tanh(h8)
        h9 = cp.dot(h8, self.W_f8) + self.b8
        h9 = cp.tanh(h9)
        h10 = cp.dot(h9, self.W_f9) + self.b9
        output = softmax(h10)
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

    def gradient(self, x, target, epoch):
        h1 = cp.dot(x, self.W_f1) + self.b1
        h1_ = cp.tanh(h1)
        h2 = cp.dot(h1_, self.W_f2) + self.b2
        h2_ = cp.tanh(h2)
        h3 = cp.dot(h2_, self.W_f3) + self.b3
        h3_ = cp.tanh(h3)
        h4 = cp.dot(h3_, self.W_f4) + self.b4
        h4_ = cp.tanh(h4)
        h5 = cp.dot(h4_, self.W_f5) + self.b5
        h5_ = cp.tanh(h5)
        h6 = cp.dot(h5_, self.W_f6) + self.b6
        h6_ = cp.tanh(h6)
        h7 = cp.dot(h6_, self.W_f7) + self.b7
        output = softmax(h7)

        delta7 = (output - target) / batch_size
        self.delta_Wf7 = cp.dot(h6_.T, delta7)
        self.delta_b7 = cp.dot(cp.ones(batch_size), delta7)

        delta6 = tanh_grad(h6) * cp.dot(delta7, self.W_f7.T)
        self.delta_Wf6 = cp.dot(h5_.T, delta6)
        self.delta_b6 = cp.dot(cp.ones(batch_size), delta6)

        delta5 = tanh_grad(h5) * cp.dot(delta6, self.W_f6.T)
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
        eta = 0.02

        self.W_f1 -= eta * self.delta_Wf1
        self.W_f2 -= eta * self.delta_Wf2
        self.W_f3 -= eta * self.delta_Wf3
        self.W_f4 -= eta * self.delta_Wf4
        self.W_f5 -= eta * self.delta_Wf5
        self.W_f6 -= eta * self.delta_Wf6
        self.W_f7 -= eta * self.delta_Wf7
        self.b1 -= eta * self.delta_b1
        self.b2 -= eta * self.delta_b2
        self.b3 -= eta * self.delta_b3
        self.b4 -= eta * self.delta_b4
        self.b5 -= eta * self.delta_b5
        self.b6 -= eta * self.delta_b6
        self.b7 -= eta * self.delta_b7

    def learning_rate(self, epoch):
        if epoch <= 20000:
            return 0.12
        elif epoch <= 30000:
            return 0.1
        elif epoch <= 50000:
            return 0.08
        elif epoch <= 100000:
            return 0.04
        elif epoch <= 200000:
            return 0.02
        else:
            return 0.015

    def feedback_alignment(self, x, target, epoch, flag):
        h1 = cp.dot(x, self.W_f1) + self.b1
        h1_ = cp.tanh(h1)
        h2 = cp.dot(h1_, self.W_f2) + self.b2
        h2_ = cp.tanh(h2)
        h3 = cp.dot(h2_, self.W_f3) + self.b3
        h3_ = cp.tanh(h3)
        h4 = cp.dot(h3_, self.W_f4) + self.b4
        h4_ = cp.tanh(h4)
        h5 = cp.dot(h4_, self.W_f5) + self.b5
        h5_ = cp.tanh(h5)
        h6 = cp.dot(h5_, self.W_f6) + self.b6
        h6_ = cp.tanh(h6)
        h7 = cp.dot(h6_, self.W_f7) + self.b7
        output = softmax(h7)

        delta7 = (output - target) / batch_size
        delta_Wf7 = cp.dot(h6_.T, delta7)
        delta_b7 = cp.dot(cp.ones(batch_size), delta7)

        delta6 = tanh_grad(h6) * cp.dot(delta7, self.B7)
        delta_Wf6 = cp.dot(h5_.T, delta6)
        delta_b6 = cp.dot(cp.ones(batch_size), delta6)

        delta5 = tanh_grad(h5) * cp.dot(delta6, self.B6)
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
        # eta = 0.02
        # calculated by back propagation
        if flag:
            delta7 = (output - target) / batch_size
            self.delta_Wf7 = cp.dot(h6_.T, delta7)
            self.delta_b7 = cp.dot(cp.ones(batch_size), delta7)

            delta6 = tanh_grad(h6) * cp.dot(delta7, self.W_f7.T)
            self.delta_Wf6 = cp.dot(h5_.T, delta6)
            self.delta_b6 = cp.dot(cp.ones(batch_size), delta6)


            delta5 = tanh_grad(h5) * cp.dot(delta6, self.W_f6.T)
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

            self.angle_W6 = self.angle(delta_Wf6, self.delta_Wf6)
            self.angle_W5 = self.angle(delta_Wf5, self.delta_Wf5)
            self.angle_W4 = self.angle(delta_Wf4, self.delta_Wf4)
            self.angle_W3 = self.angle(delta_Wf3, self.delta_Wf3)
            self.angle_W2 = self.angle(delta_Wf2, self.delta_Wf2)
            self.angle_W1 = self.angle(delta_Wf1, self.delta_Wf1)

        eta = self.learning_rate(epoch)
        self.W_f1 -= eta * delta_Wf1
        self.W_f2 -= eta * delta_Wf2
        self.W_f3 -= eta * delta_Wf3
        self.W_f4 -= eta * delta_Wf4
        self.W_f5 -= eta * delta_Wf5
        self.W_f6 -= eta * delta_Wf6
        self.W_f7 -= eta * delta_Wf7
        self.b1 -= eta * delta_b1
        self.b2 -= eta * delta_b2
        self.b3 -= eta * delta_b3
        self.b4 -= eta * delta_b4
        self.b5 -= eta * delta_b5
        self.b6 -= eta * delta_b6
        self.b7 -= eta * delta_b7

    def angle(self, A, B):
        fp = A * B
        fp = cp.sum(fp)
        norm_a = cp.sqrt(cp.sum(A * A))
        norm_b = cp.sqrt(cp.sum(B * B))
        cos_theta = fp / (norm_a * norm_b)
        return cp.arccos(cos_theta)

"""
mlp = MLP()
train_loss_list = []
test_loss_list = []
train_acc_list = []
test_acc_list = []

train_size = x_train.shape[0]
batch_size = 100
iter_per_epoch = 100
print("Back propagation")
for i in range(40000):
    batch_mask = cp.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]
    mlp.gradient(x_batch, t_batch, i)
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

np.savetxt("./result/0816/BP_cifarW1.txt", cuda.to_cpu(mlp.W_f1))
np.savetxt("./result/0816/BP_cifarW2.txt", cuda.to_cpu(mlp.W_f2))
np.savetxt("./result/0816/BP_cifarW3.txt", cuda.to_cpu(mlp.W_f3))
np.savetxt("./result/0816/BP_cifarW4.txt", cuda.to_cpu(mlp.W_f4))
np.savetxt("./result/0816/BP_cifarW5.txt", cuda.to_cpu(mlp.W_f5))
np.savetxt("./result/0816/BP_cifarb1.txt", cuda.to_cpu(mlp.b1))
np.savetxt("./result/0816/BP_cifarb2.txt", cuda.to_cpu(mlp.b2))
np.savetxt("./result/0816/BP_cifarb3.txt", cuda.to_cpu(mlp.b3))
np.savetxt("./result/0816/BP_cifarb4.txt", cuda.to_cpu(mlp.b4))
np.savetxt("./result/0816/BP_cifarb5.txt", cuda.to_cpu(mlp.b5))

"""

mlp = MLP()
train_loss_list_FA = []
test_loss_list_FA = []
train_acc_list_FA = []
test_acc_list_FA = []
f = open('./result/0820/angle_log_6layer.txt', 'a')
print("angle_Wf6, angle_Wf5, angle_Wf4, angle_Wf3, angle_Wf2, angle_Wf1", file=f)
train_size = x_train.shape[0]
batch_size = 100
iter_per_epoch = 100
print("Feedback alignment")
for i in range(100000):
    batch_mask = cp.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]
    # mlp.gradient(x_batch, t_batch)
    if i % iter_per_epoch != 0:
        mlp.feedback_alignment(x_batch, t_batch, i, False)

    if i % iter_per_epoch == 0:
        mlp.feedback_alignment(x_batch, t_batch, i, True)
        train_acc = mlp.accuracy(x_train, t_train)
        test_acc = mlp.accuracy(x_test, t_test)
        train_loss = mlp.loss(x_train, t_train)
        test_loss = mlp.loss(x_test, t_test)
        train_loss_list_FA.append(cuda.to_cpu(train_loss))
        test_loss_list_FA.append(cuda.to_cpu(test_loss))
        train_acc_list_FA.append(cuda.to_cpu(train_acc))
        test_acc_list_FA.append(cuda.to_cpu(test_acc))
        print("epoch:", int(i / iter_per_epoch), " train acc, test acc | " + str(train_acc) + ", " + str(test_acc))
        print("angle_Wf6, angle_Wf5, angle_Wf4, angle_Wf3, angle_Wf2, angle_Wf1", mlp.angle_W6, mlp.angle_W5,
              mlp.angle_W4, mlp.angle_W3,
              mlp.angle_W2, mlp.angle_W1)
        print(mlp.angle_W6, mlp.angle_W5, mlp.angle_W4, mlp.angle_W3, mlp.angle_W2, mlp.angle_W1, file=f)
f.close()


np.savetxt("./result/0820/FA_mnistW6-1.txt", cuda.to_cpu(mlp.W_f1))
np.savetxt("./result/0820/FA_mnistW6-2.txt", cuda.to_cpu(mlp.W_f2))
np.savetxt("./result/0820/FA_mnistW6-3.txt", cuda.to_cpu(mlp.W_f3))
np.savetxt("./result/0820/FA_mnistW6-4.txt", cuda.to_cpu(mlp.W_f4))
np.savetxt("./result/0820/FA_mnistW6-5.txt", cuda.to_cpu(mlp.W_f5))
np.savetxt("./result/0820/FA_mnistW6-6.txt", cuda.to_cpu(mlp.W_f6))
np.savetxt("./result/0820/FA_mnistW6-7.txt", cuda.to_cpu(mlp.W_f7))
np.savetxt("./result/0820/FA_mnistb6-1.txt", cuda.to_cpu(mlp.b1))
np.savetxt("./result/0820/FA_mnistb6-2.txt", cuda.to_cpu(mlp.b2))
np.savetxt("./result/0820/FA_mnistb6-3.txt", cuda.to_cpu(mlp.b3))
np.savetxt("./result/0820/FA_mnistb6-4.txt", cuda.to_cpu(mlp.b4))
np.savetxt("./result/0820/FA_mnistb6-5.txt", cuda.to_cpu(mlp.b5))
np.savetxt("./result/0820/FA_mnistb6-6.txt", cuda.to_cpu(mlp.b6))
np.savetxt("./result/0820/FA_mnistb6-7.txt", cuda.to_cpu(mlp.b7))

