import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np

# A typical training loop for any neural network
def training(model, epochs, train, valid, optimizer, criterion):
    for epoch in range(epochs):

        # Put model into training mode
        model.train()

        for data, labels in train:
            # get the batch you are training with
            inputs = data.to(device)
            labels = labels.to(device)

            # clear previous gradients
            optimizer.zero_grad()

            # get the output layer, or predictions
            outputs = model(inputs)

            # calculate the loss
            loss = criterion(outputs, labels)

            # Backprop is very simple in pytorch. Two lines of code. Just get the gradients from the loss using loss.backward(), and then update your optimizer with .step().
            loss.backward()
            optimizer.step()
