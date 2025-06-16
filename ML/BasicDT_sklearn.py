import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

filepath = r"..." # replace with file path here

data = pd.read_csv(filepath)

targ = data.label.values
feat = data.loc[:, data.columns != "label"]

Xt, Xv, Yt, Yv = train_test_split(feat, targ, test_size = 0.8, random_state=24)

clf = DecisionTreeClassifier(random_state=24)
clf.fit(Xt, Yt)

y_pred = clf.predict(Xv)
acc = accuracy_score(Yv, y_pred)

print(f'accuracy: {acc}')
