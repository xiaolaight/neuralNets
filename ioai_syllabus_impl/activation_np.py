import numpy as np

#ReLU

def ReLU(x):
    return np.maximum(x, 0)

def deriv_ReLU(x):
    return x > 0

# sigmoid

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def deriv_sigmoid(x):
    sx = sigmoid(x)
    return sx * (1-sx)

# tanh

def tanh(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

def deriv_tanh(x):
    return 1 - tanh(x)*tanh(x)

# softmax

def softmax(x):
    x_max = np.max(x, axis=0, keepdims=True)
    e_x = np.exp(x - x_max)
    S = np.sum(e_x, axis=0, keepdims=True)
    return e_x / S
