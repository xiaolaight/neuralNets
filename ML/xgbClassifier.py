import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
import xgboost as xgb

def get_mx(A):
    x = np.array(A).T
    x = np.argmax(x, 0)
    return x

def getAcc(x, y):
    return np.sum(y == x) / x.size

param = {
    'eta': 0.3,
    'max_depth': 15,
    'num_class': 10,
    'n_estimators': 100
}

filepath = r"..." #replace with filepath

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
