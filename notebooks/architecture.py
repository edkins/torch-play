from typing import Sequence
from numpy import size
import torch

from layers import ConvLayer, DenseLayer, FlattenLayer, ReluLayer

class LayeredModule(torch.nn.Module):
    def __init__(self, *layers: Sequence[torch.nn.Module]):
        super().__init__()
        self.layers = torch.nn.ModuleList(layers)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return self.softmax(x)

    def forward_steps(self, x: torch.Tensor) -> list[torch.Tensor]:
        result = [x.detach()]
        for layer in self.layers:
            x = layer(x)
            result.append(x.detach())
        return result

    def __str__(self):
        return self.__class__.__name__

class LinearSoftmax(LayeredModule):
    def __init__(self, in_w: int, in_h: int, in_channels: int, out_features: int):
        size = (in_channels, in_w, in_h)
        flatten = FlattenLayer(size)
        linear = DenseLayer(flatten.size, out_features)
        super().__init__(flatten, linear)

class CNN1(LayeredModule):
    def __init__(self, in_w: int, in_h: int, in_channels: int, out_features: int):
        super().__init__()
        size = (in_channels, in_w, in_h)
        conv1 = ConvLayer(size, out_channels=6, kernel_size=3, stride=1)
        relu1 = ReluLayer(conv1.size)
        flatten = FlattenLayer(relu1.size)
        dense1 = DenseLayer(flatten.size, out_features)
        super().__init__(conv1, relu1, flatten, dense1)

class CNN2(LayeredModule):
    def __init__(self, in_w: int, in_h: int, in_channels: int, out_features: int):
        super().__init__()
        size = (in_channels, in_w, in_h)
        conv1 = ConvLayer(size, out_channels=3, kernel_size=3, stride=1)
        relu1 = ReluLayer(conv1.size)
        flatten = FlattenLayer(relu1.size)
        dense1 = DenseLayer(flatten.size, out_features)
        super().__init__(conv1, relu1, flatten, dense1)
