import numpy as np

def deriv_ReLU(x):
    return x > 0

def backward(wo, bo, wt, bt, a1, a2, z1, z2, X, Y, alpha):
    # Reverses the mean squared error cost function
    dz2 = a2 - Y
    # Adjusting the weights. Each weight is multiplied by the prior activated layer, so taking the dot product of the transpose reverses this.
    wt -= alpha * 1/rows * dz2.dot(a1.T)
    # Adjusting the biases. All of the biases affect the end dz2 product by being added to it. Therefore, subtracting the average of the biases will reverse this.
    bt -= alpha * 1/rows * np.sum(dz2)
    # This is the gradient of the first layer. ReLU is applied to this layer, which is then fed into weights which change the gradient of the second layer. The below reverses this process.
    dz1 = wt.T.dot(dz2) * deriv_ReLU(z1)
    # Similar to above wt
    wo -= alpha * 1/rows * dz1.dot(X.T)
    # Similar to above bt
    bo -= alpha * 1/rows * np.sum(dz1)
    return wo, bo, wt, bt
