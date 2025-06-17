# ~97% accuracy achieved, very good baseline which CNN can improve on

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
import xgboost as xgb

def get_mx(A):
    x = np.array(A).T
    x = np.argmax(x, 0)
    return x

def getAcc(x, y):
    return np.sum(y == x) / x.size

def showErr(X, Y, ans):
    cor = np.equal(Y, ans)
    for i in range(len(cor)):
        if cor[i] == False:
            print(f"Predicted: {ans[i]}   Shown: {Y[i]}")
            plt.imshow(np.array(X.loc[X.index[i]]).reshape(28, 28) * 255)
            plt.show()
            break

param = {
    'eta': 0.3,
    'max_depth': 15,
    'num_class': 10,
    'n_estimators': 100
}

filepath = r"..."

data = pd.read_csv(filepath)

targ = data.label.values
feat = data.loc[:, data.columns != "label"]

Xt, Xv, Yt, Yv = train_test_split(feat, targ, test_size=0.2, random_state=24)

clf = xgb.XGBClassifier(objective='multi:softprob', **param)
clf.fit(Xt, Yt)

y_pred = clf.predict_proba(Xv)
y_pred = get_mx(y_pred)
print(y_pred)
print(Yv)
acc = getAcc(Yv, y_pred)

print(f'accuracy: {acc}')

showErr(Xv, Yv, y_pred)
