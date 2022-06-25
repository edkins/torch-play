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

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using {device} device")

_, _, w, h = next(iter(train_dataloader))[0].size()
print(f"w = {w}, h = {h}")

class NeuralNetwork(nn.Module):
    def __init__(self, w, h):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
                nn.Linear(w * h, 512),
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork(w,h).to(device)
print(model)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    # Set to training mode
    model.train()
    for batch_num, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch_num % 100 == 0:
            loss, current = loss.item(), batch_num * len(X)
            print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return correct

epochs = 10
correctness = []
for t in range(epochs):
    print(f"Epoch {t+1}\n----------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    correct = test(test_dataloader, model, loss_fn)
    correctness.append(correct)
    plt.clf()
    plt.plot(correctness)
    plt.ylabel('accuracy')
    plt.ylim((0, 1))
    plt.xlabel('epoch')
    plt.pause(0.1)
plt.show()
print("Done!")

