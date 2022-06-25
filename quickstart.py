import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

import matplotlib.pyplot as plt
import math

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
        self.display = False
        self.transforms = nn.ModuleList([
                nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2),
                nn.ReLU(),
                nn.AvgPool2d(kernel_size=2, stride=2),
                nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5),
                nn.ReLU(),
                nn.AvgPool2d(kernel_size=2, stride=2),
                nn.Flatten(),
                nn.Linear(5*5*16, 120),
                nn.ReLU(),
                nn.Linear(120, 84),
                nn.ReLU(),
                nn.Linear(84, 10),
        ])
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        for i,t in enumerate(self.transforms):
            x = t(x)
            if self.display:
                plt.clf()
                for j in range(4):
                    if len(x.size())== 2:
                        width = math.ceil(math.sqrt(x.size()[1]))
                        height = width
                        extra = torch.zeros(width * height - len(x[j])).to(device)
                        img = torch.cat((x[j], extra)).reshape((width,height)).cpu().detach()
                        plt.subplot(4, 1, 1 + j)
                        plt.xticks(ticks=[])
                        plt.yticks(ticks=[])
                        plt.imshow(img)
                    else:
                        _, count, width, height = x.size()
                        for k in range(count):
                            plt.subplot(4, count, 1 + k + j * count)
                            plt.xticks(ticks=[])
                            plt.yticks(ticks=[])
                            img = x[j][k].cpu().detach()
                            plt.imshow(img)
                print(i, x.size(), t)
                plt.show()
        self.display = False
        return self.softmax(x)

model = NeuralNetwork(w,h).to(device)
print(model)

loss_fn = nn.CrossEntropyLoss(reduction='sum')
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
    #model.display = (t == 1)
    print(list(model.parameters()))
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

