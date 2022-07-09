import torch

from layers import ConvLayer, DenseLayer, FlattenLayer, ReluLayer

class LinearSoftmax(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = torch.nn.Linear(in_features, out_features)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.softmax(self.linear(x.flatten(1)))

    def __str__(self):
        return 'LinearSoftmax'

class CNN1(torch.nn.Module):
    def __init__(self, in_w: int, in_h: int, in_channels: int, out_features: int):
        super().__init__()
        size = (in_channels, in_w, in_h)
        self.conv1 = ConvLayer(size, out_channels=6, kernel_size=3, stride=1)
        self.relu1 = ReluLayer(self.conv1.size)
        self.flatten = FlattenLayer(self.relu1.size)
        self.dense1 = DenseLayer(self.flatten.size, out_features)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.flatten(x)
        x = self.dense1(x)
        return self.softmax(x)
