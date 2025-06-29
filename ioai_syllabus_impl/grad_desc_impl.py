import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import numpy as np
import pandas as pd

import matplotlib as plt
from torchsummary import summary

from sklearn.model_selection import train_test_split

# The above libraries are standard libraries when it comes to managing data

# The CustomDataset Class essentially transforms your data into a format which pytorch can work with
# You can rename the class to whatever

class CustomDataset(Dataset):
    # This initializes your data and labels
    # Transform and shuffle are not important for now
    def __init__(self, data, labels, transform=None, shuffle=True):
        self.data = data.astype(np.float32)
        self.labels = labels.astype(np.int64)
        self.transform = transform

    # Gives your dataset object a length attribute, which is just how many datapoints there are
    def __len__(self):
        return len(self.data)

    # To give your dataset the ability to be accessed and indexed
    def __getitem__(self, idx):
        data = self.data[idx]
        label = self.labels[idx]
        if self.transform:
            data = self.transform(data)
        return data, label

filepath = r"..." # modify to your filepath with your data
df = pd.read_csv(filepath, dtype = np.float32)

targ = df.label.values
feat = df.iloc[:, df.columns != "label"].values

X_train, X_valid, Y_train, Y_valid = train_test_split(feat, targ, test_size = 0.2, random_state=24)

# Here is the mini-batch gradient descent. The batch size is the last parameter.
# Larger batch sizes are more memory intensive but fast for training as more variables are vectorized at once.
# It can lead to overfitting however, and it can fail to escape a local minima (think of it as less chances to escape).
# Smaller batch sizes are less memory intensive but slower for training.
# They generalize very easily, but suffer slow training and performance loss because there are so many groups of data.
train = DataLoader(CustomDataset(X_train, Y_train, shuffle=True), 100)
valid = DataLoader(CustomDataset(X_valid, Y_valid, shuffle=False), 100)
