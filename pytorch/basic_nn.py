import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

import numpy as np
import pandas as pd

import matplotlib as plt
from torchsummary import summary

from sklearn.model_selection import train_test_split

class MNIST_ReLU(nn.Module):
    def __init__(self):
        super(MNIST_ReLU, self).__init__()
        self.lay = nn.Sequential(
            nn.Linear(784, 24, bias=True),
            nn.ReLU(),
            nn.Linear(24, 10)
        )

    def forward(self, x):
        x = self.lay(x)
        return x

class CustomDataset(Dataset):
    def __init__(self, data, labels, transform=None, shuffle=True):
        self.data = data.astype(np.float32)
        self.labels = labels.astype(np.int64)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        label = self.labels[idx]
        if self.transform:
            data = self.transform(data)
        return data, label

def validate(model, valid, criterion, device='cpu'):
    model.eval()
    loss = 0.0
    cor = 0
    samples = 0

    with torch.no_grad():
        for data, labels in valid:
            inp = data.to(device)
            lab = labels.to(device)

            val_outputs = model(inp)
            val_loss = criterion(val_outputs, lab)

            loss += val_loss.item()

            value_waste, pred = torch.max(val_outputs, 1)
            cor += (pred == lab).sum().item()
            samples += lab.size(0)

    validation_accuracy = cor / samples
    return loss / len(valid), validation_accuracy * 100

def training(model, epochs, train, valid, optimizer, criterion):
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        correct_train_predictions = 0
        samples = 0

        for data, labels in train:
            inputs = data.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)

            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            value_waste, pred = torch.max(outputs, 1)
            correct_train_predictions += (pred == labels).sum().item()
            samples += labels.size(0)

        training_accuracy = correct_train_predictions / samples

        print(f'Epoch [{epoch + 1}/{epochs}], Training Loss: {total_loss / len(train):.3f}, Training Accuracy: {training_accuracy * 100:.3f}%')
        if valid is not None:
            val_loss, val_accuracy = validate(model, valid, criterion,device=device)
            print(f'Epoch [{epoch + 1}/{epochs}], Validation Loss: {val_loss:.3f}, Validation Accuracy: {val_accuracy:.3f}%\n')

    print('Training finished')

df = pd.read_csv(r"C:\Users\andig\Downloads\train.csv\train.csv", dtype = np.float32)

targ = df.label.values
feat = df.iloc[:, df.columns != "label"].values / 255.0

X_train, X_valid, Y_train, Y_valid = train_test_split(feat, targ, test_size = 0.2, random_state=24)

train = DataLoader(CustomDataset(X_train, Y_train, shuffle=True), 100)
valid = DataLoader(CustomDataset(X_valid, Y_valid, shuffle=False), 100)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = MNIST_ReLU().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
epochs = 30

training(model, epochs, train, valid, optimizer, criterion)
