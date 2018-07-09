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

        self.allones = weight_init_std * cp.ones([10, hidden_unit])

        self.d = np.random.randn(10)
        # print(d)
        self.B3_ll = []
        for i in range(1000):
            magnification = np.random.rand() * 2 - 1
            self.B3_ll.append(self.d * magnification)
        self.B3_ll = weight_init_std * cp.array(self.B3_ll)
        self.B3_ll = self.B3_ll.T
        self.B2_ll = []
        for i in range(1000):
            magnification = np.random.rand() * 2 - 1
            self.B2_ll.append(self.d * magnification)
        self.B2_ll = weight_init_std * cp.array(self.B2_ll)
        self.B2_ll = self.B2_ll.T
        self.B1_ll = []
        for i in range(1000):
            magnification = np.random.rand() * 2 - 1
            self.B1_ll.append(self.d * magnification)
        self.B1_ll = weight_init_std * cp.array(self.B1_ll)
        self.B1_ll = self.B1_ll.T

        self.B3_ll3 = []
        for i in range(1000):
            # magnification = np.random.rand() * 2 - 1
            self.B3_ll3.append(self.d)
        self.B3_ll3 = weight_init_std * cp.array(self.B3_ll3)
        self.B3_ll3 = self.B3_ll3.T
        self.B2_ll3 = []
        for i in range(1000):
            # magnification = np.random.rand() * 2 - 1
            self.B2_ll3.append(self.d)
        self.B2_ll3 = weight_init_std * cp.array(self.B2_ll3)
        self.B2_ll3 = self.B2_ll3.T
        self.B1_ll3 = []
        for i in range(1000):
            # magnification = np.random.rand() * 2 - 1
            self.B1_ll3.append(self.d)
        self.B1_ll3 = weight_init_std * cp.array(self.B1_ll3)
        self.B1_ll3 = self.B1_ll3.T

        ones = np.ones(10)
        self.B3_ll2 = []
        for i in range(1000):
            magnification = np.random.rand() * 2 - 1
            self.B3_ll2.append(ones * magnification)
        self.B3_ll2 = weight_init_std * cp.array(self.B3_ll2)
        self.B3_ll2 = self.B3_ll2.T
        self.B2_ll2 = []
        for i in range(1000):
            magnification = np.random.rand() * 2 - 1
            self.B2_ll2.append(ones * magnification)
        self.B2_ll2 = weight_init_std * cp.array(self.B2_ll2)
        self.B2_ll2 = self.B2_ll2.T
        self.B1_ll2 = []
        for i in range(1000):
            magnification = np.random.rand() * 2 - 1
            self.B1_ll2.append(ones * magnification)
        self.B1_ll2 = weight_init_std * cp.array(self.B1_ll2)
        self.B1_ll2 = self.B1_ll2.T

        self.B3_dfa = weight_init_std * cp.random.randn(10, hidden_unit)
        self.B2_dfa = weight_init_std * cp.random.randn(10, hidden_unit)
        self.B1_dfa = weight_init_std * cp.random.randn(10, hidden_unit)

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

    def direct_feedback_alignment(self, x, target):
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

        delta3 = cp.dot(delta4, self.B3_dfa)
        delta_Wf3 = cp.dot(h2_.T, relu_grad(h3) * delta3)

        delta2 = cp.dot(delta4, self.B2_dfa)
        delta_Wf2 = cp.dot(h1_.T, relu_grad(h2) * delta2)

        delta1 = cp.dot(delta4, self.B1_dfa)
        delta_Wf1 = cp.dot(x.T, relu_grad(h1) * delta1)

        alpha1 = 0.1
        self.W_f1 -= alpha1 * delta_Wf1
        self.W_f2 -= alpha1 * delta_Wf2
        self.W_f3 -= alpha1 * delta_Wf3
        self.W_f4 -= alpha1 * delta_Wf4

    def unified_global_error(self, x, target):
        h1 = cp.dot(x, self.W_f1)
        h1_ = relu(h1)
        h2 = cp.dot(h1_, self.W_f2)
        h2_ = relu(h2)
        h3 = cp.dot(h2_, self.W_f3)
        h3_ = relu(h3)
        h4 = cp.dot(h3_, self.W_f4)
        output = softmax(h4)

        self.delta4 = (output - target) / batch_size
        delta_Wf4 = cp.dot(h3_.T, self.delta4)

        delta3 = cp.dot(self.delta4, self.allones)
        delta_Wf3 = cp.dot(h2_.T, relu_grad(h3) * delta3)

        delta2 = cp.dot(self.delta4, self.allones)
        delta_Wf2 = cp.dot(h1_.T, relu_grad(h2) * delta2)

        delta1 = cp.dot(self.delta4, self.allones)
        delta_Wf1 = cp.dot(x.T, relu_grad(h1) * delta1)

        alpha1 = 0.1
        self.W_f1 -= alpha1 * delta_Wf1
        self.W_f2 -= alpha1 * delta_Wf2
        self.W_f3 -= alpha1 * delta_Wf3
        self.W_f4 -= alpha1 * delta_Wf4

    def local_learning_rule(self, x, target):
        h1 = cp.dot(x, self.W_f1)
        h1_ = relu(h1)
        h2 = cp.dot(h1_, self.W_f2)
        h2_ = relu(h2)
        h3 = cp.dot(h2_, self.W_f3)
        h3_ = relu(h3)
        h4 = cp.dot(h3_, self.W_f4)
        output = softmax(h4)

        self.delta4 = (output - target) / batch_size
        delta_Wf4 = cp.dot(h3_.T, self.delta4)

        delta3 = cp.dot(self.delta4, self.B3_ll)
        delta_Wf3 = cp.dot(h2_.T, relu_grad(h3) * delta3)

        delta2 = cp.dot(self.delta4, self.B2_ll)
        delta_Wf2 = cp.dot(h1_.T, relu_grad(h2) * delta2)

        delta1 = cp.dot(self.delta4, self.B1_ll)
        delta_Wf1 = cp.dot(x.T, relu_grad(h1) * delta1)

        alpha1 = 0.1
        self.W_f1 -= alpha1 * delta_Wf1
        self.W_f2 -= alpha1 * delta_Wf2
        self.W_f3 -= alpha1 * delta_Wf3
        self.W_f4 -= alpha1 * delta_Wf4

    def local_learning_rule2(self, x, target):
        h1 = cp.dot(x, self.W_f1)
        h1_ = relu(h1)
        h2 = cp.dot(h1_, self.W_f2)
        h2_ = relu(h2)
        h3 = cp.dot(h2_, self.W_f3)
        h3_ = relu(h3)
        h4 = cp.dot(h3_, self.W_f4)
        output = softmax(h4)

        self.delta4 = (output - target) / batch_size
        delta_Wf4 = cp.dot(h3_.T, self.delta4)

        delta3 = cp.dot(self.delta4, self.B3_ll2)
        delta_Wf3 = cp.dot(h2_.T, relu_grad(h3) * delta3)

        delta2 = cp.dot(self.delta4, self.B2_ll2)
        delta_Wf2 = cp.dot(h1_.T, relu_grad(h2) * delta2)

        delta1 = cp.dot(self.delta4, self.B1_ll2)
        delta_Wf1 = cp.dot(x.T, relu_grad(h1) * delta1)

        alpha1 = 0.1
        self.W_f1 -= alpha1 * delta_Wf1
        self.W_f2 -= alpha1 * delta_Wf2
        self.W_f3 -= alpha1 * delta_Wf3
        self.W_f4 -= alpha1 * delta_Wf4

    def local_learning_rule3(self, x, target):
        h1 = cp.dot(x, self.W_f1)
        h1_ = relu(h1)
        h2 = cp.dot(h1_, self.W_f2)
        h2_ = relu(h2)
        h3 = cp.dot(h2_, self.W_f3)
        h3_ = relu(h3)
        h4 = cp.dot(h3_, self.W_f4)
        output = softmax(h4)

        self.delta4 = (output - target) / batch_size
        delta_Wf4 = cp.dot(h3_.T, self.delta4)

        delta3 = cp.dot(self.delta4, self.B3_ll3)
        delta_Wf3 = cp.dot(h2_.T, relu_grad(h3) * delta3)

        delta2 = cp.dot(self.delta4, self.B2_ll3)
        delta_Wf2 = cp.dot(h1_.T, relu_grad(h2) * delta2)

        delta1 = cp.dot(self.delta4, self.B1_ll3)
        delta_Wf1 = cp.dot(x.T, relu_grad(h1) * delta1)

        alpha1 = 0.1
        self.W_f1 -= alpha1 * delta_Wf1
        self.W_f2 -= alpha1 * delta_Wf2
        self.W_f3 -= alpha1 * delta_Wf3
        self.W_f4 -= alpha1 * delta_Wf4

    def angle(self, a, b):
        a = cuda.to_cpu(a)
        b = cuda.to_cpu(b)
        # norm_a = []
        norm_b = []
        theta = []
        inner_pro = []
        for i in range(b.shape[0]):
            tmp = np.dot(a, b[i])/(np.linalg.norm(a)*np.linalg.norm(b[i]))
            norm_b.append(np.linalg.norm(b[i]))
            inner_pro.append(np.dot(a, b[i]))
            theta.append(np.arccos(tmp))
        norm_b = np.array(norm_b)
        theta = np.array(theta)
        inner_pro = np.array(inner_pro)
        return np.mean(inner_pro), np.linalg.norm(a)*0.01, np.mean(norm_b), np.mean(theta)


mlp = MLP()
test_acc_list_ll3 = []
# print("direct feedback alignment")
print("local learning rule3")
train_size = x_train.shape[0]
batch_size = 100
iter_per_epoch = 100
for i in range(100000):
    batch_mask = cp.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]
    mlp.local_learning_rule3(x_batch, t_batch)
    if i % iter_per_epoch == 0:
        train_acc = mlp.accuracy(x_train, t_train)
        test_acc = mlp.accuracy(x_test, t_test)
        train_loss = mlp.loss(x_train, t_train)
        test_loss = mlp.loss(x_test, t_test)
        test_acc_list_ll3.append(cuda.to_cpu(test_acc))
        # print("epoch:", int(i / iter_per_epoch), " train loss, test loss, train acc, test acc | " + str(train_loss)
        #       + ", " + str(test_loss) + ", " + str(train_acc) + ", " + str(test_acc))
        # print(mlp.d)
        # print(mlp.delta4)
        # print(cp.dot(cp.asarray(mlp.d), mlp.delta4))
        print(int(i / iter_per_epoch), train_loss, mlp.angle(cp.asarray(mlp.d), mlp.delta4))

mlp = MLP()
test_acc_list_uge = []
print("unified global error")
train_size = x_train.shape[0]
batch_size = 100
iter_per_epoch = 100
for i in range(100000):
    batch_mask = cp.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]
    mlp.unified_global_error(x_batch, t_batch)
    if i % iter_per_epoch == 0:
        train_acc = mlp.accuracy(x_train, t_train)
        test_acc = mlp.accuracy(x_test, t_test)
        train_loss = mlp.loss(x_train, t_train)
        test_loss = mlp.loss(x_test, t_test)
        test_acc_list_uge.append(cuda.to_cpu(test_acc))
        print(int(i / iter_per_epoch), train_loss, mlp.angle(cp.ones(10), mlp.delta4))
        # print("epoch:", int(i / iter_per_epoch), " train loss, test loss, train acc, test acc | " + str(train_loss)
        #       + ", " + str(test_loss) + ", " + str(train_acc) + ", " + str(test_acc))

mlp = MLP()
test_acc_list_ll = []
print("local learning")
print(mlp.d)
train_size = x_train.shape[0]
batch_size = 100
iter_per_epoch = 100
for i in range(100000):
    batch_mask = cp.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]
    mlp.local_learning_rule(x_batch, t_batch)
    if i % iter_per_epoch == 0:
        train_acc = mlp.accuracy(x_train, t_train)
        test_acc = mlp.accuracy(x_test, t_test)
        train_loss = mlp.loss(x_train, t_train)
        test_loss = mlp.loss(x_test, t_test)
        test_acc_list_ll.append(cuda.to_cpu(test_acc))
        print(int(i / iter_per_epoch), train_loss, mlp.angle(cp.asarray(mlp.d), mlp.delta4))
        # print("epoch:", int(i / iter_per_epoch), " train loss, test loss, train acc, test acc | " + str(train_loss)
        #       + ", " + str(test_loss) + ", " + str(train_acc) + ", " + str(test_acc))

mlp = MLP()
test_acc_list_ll2 = []
print("local learning2")
train_size = x_train.shape[0]
batch_size = 100
iter_per_epoch = 100
for i in range(100000):
    batch_mask = cp.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]
    mlp.local_learning_rule2(x_batch, t_batch)
    if i % iter_per_epoch == 0:
        train_acc = mlp.accuracy(x_train, t_train)
        test_acc = mlp.accuracy(x_test, t_test)
        train_loss = mlp.loss(x_train, t_train)
        test_loss = mlp.loss(x_test, t_test)
        test_acc_list_ll2.append(cuda.to_cpu(test_acc))
        print(int(i / iter_per_epoch), train_loss, mlp.angle(cp.ones(10), mlp.delta4))
        # print("epoch:", int(i / iter_per_epoch), " train loss, test loss, train acc, test acc | " + str(train_loss)
        #       + ", " + str(test_loss) + ", " + str(train_acc) + ", " + str(test_acc))

plt.figure()
plt.plot(test_acc_list_ll3, label="local learning rule3", color="crimson")
# plt.plot(test_acc_list_uge, label="unified global error", color="darkblue")
# plt.plot(test_acc_list_ll, label="local learning rule1", color="forestgreen")
# plt.plot(test_acc_list_ll2, label="local learning rule2", color="gold")

plt.title("test accuracy for MNIST")
plt.xlabel("epoch")
plt.ylabel("acc")
plt.legend()

os.makedirs('./result/0709/', exist_ok=True)
plt.savefig("./result/0709/local_learning_rule3.png")
# plt.savefig("mnistBP.png")


