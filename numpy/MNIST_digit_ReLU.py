import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

def backward(wo, bo, wt, bt, a1, a2, z1, z2, X, Y, alpha):
    dz2 = a2 - Y
    wt -= alpha * 1/rows * dz2.dot(a1.T)
    bt -= alpha * 1/rows * np.sum(dz2)
    dz1 = wt.T.dot(dz2) * deriv_ReLU(z1)
    wo -= alpha * 1/rows * dz1.dot(X.T)
    bo -= alpha * 1/rows * np.sum(dz1)
    return wo, bo, wt, bt

def grad(X, Y, alpha, epochs):
    wo, bo, wt, bt = init()
    save = 0
    for i in range(epochs):
        z1, a1, z2, a2 = forward(wo, bo, wt, bt, X)
        wo, bo, wt, bt = backward(wo, bo, wt, bt, a1, a2, z1, z2, X, Y, alpha)
        save = a2
    return save

def showErr(X, Y, ans):
    test = np.argmax(ans, 0)
    vals = np.argmax(Y, 0)
    print(np.sum(test == vals) / rows)
    cor = np.equal(test, vals)
    for i in range(sep):
        if cor[i] == False:
            print(test[i])
            plt.imshow(X[:, i].reshape(28, 28) * 255)
            plt.show()
            break

# customize to path of your CSV (just click copy path and paste, dealing with backslashes when necessary
filename = "..."

df = pd.read_csv(filename)

sep = 40000

rows, cols = df.shape

train_set = df[0:sep]

X_train = train_set.drop(columns=["label"]) / 255.0
X_train = np.array(X_train).T
Y_train = np.array(train_set["label"]).T
Y_train = one_hot(Y_train)

preds = grad(X_train, Y_train, 0.08, 800)

showErr(X_train, Y_train, preds)
