import numpy as np
import chainer
import cupy as cp


# Load the fashion-MNIST dataset
train, test = chainer.datasets.get_fashion_mnist()
x_train, t_train = train._datasets
x_test, t_test = test._datasets

x_train = cp.asarray(x_train)
x_test = cp.asarray(x_test)

t_train = cp.identity(10)[t_train.astype(int)]
t_test = cp.identity(10)[t_test.astype(int)]


W_f1 = cp.load('weights_dfa_relu_W_f1.npy')
W_f2 = cp.load('weights_dfa_relu_W_f2.npy')
W_f3 = cp.load('weights_dfa_relu_W_f3.npy')
W_f4 = cp.load('weights_dfa_relu_W_f4.npy')
W_f5 = cp.load('weights_dfa_relu_W_f5.npy')

x = x_train
h1 = cp.dot(x, W_f1)
h = cp.tanh(h1)
h2 = cp.dot(h, W_f2)
h = cp.tanh(h2)
h3 = cp.dot(h, W_f3)
h = cp.tanh(h3)
h4 = cp.dot(h, W_f4)

print(cp.var(h1))
print(cp.var(h2))
print(cp.var(h3))
print(cp.var(h4))
