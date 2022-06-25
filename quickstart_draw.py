import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

import matplotlib.pyplot as plt

def load_data(train):
    return datasets.FashionMNIST(root='data', train=train, download=True, transform=ToTensor())

def show_images(X, y):
    nrows = 16
    ncols = 16
    counts = [0] * nrows
    for i in range(X.shape[0]):
        row = y[i].item()
        col = counts[row]
        counts[row] += 1
        img = X[i][0]

        if row < nrows and col < ncols:
            plt.subplot(nrows, ncols, 1 + col + row * nrows)
            plt.imshow(img, cmap='Greys')
    plt.show()

training_data = load_data(True)
test_data = load_data(False)

batch_size = 64

train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

for X, y in test_dataloader:
    show_images(X, y)
    break
