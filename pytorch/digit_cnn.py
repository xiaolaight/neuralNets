# 99.3% accuracy locally, if submitting to kaggle, then reduce the number of convolutional layers to prevent data overfitting.
# This is also on relatively fast training settings.
# You can increase padding and filter sizes to perhaps improve performance at the cost of more training.
# If you want to test this locally, just use train_test_split on the targ and feat variables to create your cross-validation set.

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

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

def calculate_max_pool_output_shape(input_height, input_width, pool_size=2):

    return int(input_height / pool_size), int(input_width / pool_size)

def find_conv2d_output_shape(height, width, conv):
    kernel_size = conv.kernel_size
    stride = conv.stride
    padding = conv.padding
    dilation = conv.dilation
    height = np.floor((height + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) / stride[0] + 1)
    width = np.floor((width + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) / stride[1] + 1)

    return int(height), int(width)

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
            data = torch.from_numpy(data)
            data = self.transform(data)
        return data, label

class MNIST_cnn(nn.Module):
    def __init__(self):
        super(MNIST_cnn, self).__init__()
        c = 1
        h = 28
        w = 28
        classes = 10

        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2)

        self.conv1 = nn.Conv2d(c, 32, kernel_size=3)
        h, w = find_conv2d_output_shape(h, w, self.conv1)
        h, w = calculate_max_pool_output_shape(h, w, 2)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3)
        h, w = find_conv2d_output_shape(h, w, self.conv2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3)
        h, w = find_conv2d_output_shape(h, w, self.conv3)
        h, w = calculate_max_pool_output_shape(h, w, 2)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3)
        h, w = find_conv2d_output_shape(h, w, self.conv4)
        h, w = calculate_max_pool_output_shape(h, w, 2)

        self.flatten = nn.Flatten()
        self.norm = nn.BatchNorm1d(h*w*64)

        self.linear1 = nn.Linear(h * w * 64, 128)
        self.drop = nn.Dropout(p=0.5)
        self.linear2 = nn.Linear(128, classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.conv2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.conv4(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.flatten(x)
        x = self.norm(x)

        x = self.linear1(x)
        x = self.relu(x)
        x = self.drop(x)
        x = self.linear2(x)

        return x


transform = v2.Compose([
    v2.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1))
])

device = 'cuda' if torch.cuda.is_available() else 'cpu'

testpath = r"..."
test = torch.from_numpy(np.array(pd.read_csv(testpath, dtype=np.float32)))
test = test.reshape(-1, 1, 28, 28)
samples = test.size(dim=0) + 1
test = test.to(device)

filepath = r"..."
df = pd.read_csv(filepath, dtype = np.float32)

targ = df.label.values
feat = df.iloc[:, df.columns != "label"].values / 255.0

X_train, X_valid, Y_train, Y_valid = train_test_split(feat, targ, test_size=0.2, random_state=24)

feat = feat.reshape(-1, 1, 28, 28)
X_train = X_train.reshape(-1, 1, 28, 28)
X_valid = X_valid.reshape(-1, 1, 28, 28)

train = DataLoader(CustomDataset(feat, targ, transform=transform, shuffle=True), 100)

model = MNIST_cnn().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005)
epochs = 30

training(model, epochs, train, None, optimizer, criterion)

_, pred = torch.max(model(test), 1)
col1 = ["ImageId"] + np.arange(1, samples, dtype=int).tolist()
col2 = ["Label"] + np.array(pred, dtype=int).tolist()
np.savetxt("results.csv", [p for p in zip(col1, col2)], delimiter=',', fmt='%s')
