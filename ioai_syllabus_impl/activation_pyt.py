import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# ReLU
nn.ReLU()
nn.LeakyReLU() # deal with vanishing gradient better, and to avoid shutting down nodes when they become negative

# sigmoid
nn.Sigmoid()

# tanh
nn.Tanh()

# more activation functions can be found here: https://docs.pytorch.org/docs/stable/nn.html
