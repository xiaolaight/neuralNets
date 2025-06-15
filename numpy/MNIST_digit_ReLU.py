import numpy as np
import pandas as pd

def ReLU(x):
    return np.maximum(x, 0)

def deriv_ReLU(x):
    return x > 0

def softmax(x):
    x_max = np.max(x, axis=0, keepdims=True)
    e_x = np.exp(x - x_max)
    S = np.sum(e_x, axis=0, keepdims=True)
    return e_x / S

def one_hot(x):
    ret = np.zeros((x.size, x.max()+1))
    ret[np.arange(x.size), x] = 1
    ret = ret.T
    return ret

def init():
    wo = np.random.rand(10, 784) - 0.5
    bo = np.random.rand(10, 1) - 0.5
    wt = np.random.rand(10, 10) - 0.5
    bt = np.random.rand(10, 1) - 0.5
    return wo, bo, wt, bt

def forward(wo, bo, wt, bt, X):
    nxt = wo.dot(X) + bo
    active = ReLU(nxt)
    nxt2 = wt.dot(active) + bt
    active2 = softmax(nxt2)
    return nxt, active, nxt2, active2

def backward(wt, a1, a2, z1, X, Y):
    encoded = one_hot(Y)
    dz2 = a2 - encoded
    dw2 = 1/rows * dz2.dot(a1.T)
    db2 = 1/rows * np.sum(dz2)
    dz1 = wt.T.dot(dz2) * deriv_ReLU(z1)
    dw1 = 1/rows * dz1.dot(X.T)
    db1 = 1/rows * np.sum(dz1)
    return dw1, dw2, db1, db2

def update(alpha, wo, bo, wt, bt, dw1, dw2, db1, db2):
    wo -= alpha*dw1
    bo -= alpha*db1
    wt -= alpha*dw2
    bt -= alpha*db2
    return wo, bo, wt, bt

def gradient_descent(X, Y, alpha, epochs):
    wo, bo, wt, bt = init()
    for i in range(epochs):
        z1, a1, z2, a2 = forward(wo, bo, wt, bt, X)
        dw1, dw2, db1, db2 = backward(wt, a1, a2, z1, X, Y)
        wo, bo, wt, bt = update(alpha, wo, bo, wt, bt, dw1, dw2, db1, db2)
        if i % 10 == 0:
            test = np.argmax(a2, 0)
            print(np.sum(test == Y) / rows)

filename = "..." # customize to path of your CSV (just click copy path and paste, dealing with backslashes when necessary

df = pd.read_csv(filename)
df = np.array(df)
np.random.shuffle(df)

sep = 40000

rows, cols = df.shape

train_set = df[0:sep].T

X_train = train_set[1:cols]
X_train = X_train / 255.0
Y_train = train_set[0]

gradient_descent(X_train, Y_train, 0.1, 5000)
