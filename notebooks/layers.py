import torch

class ConvLayer(torch.nn.Conv2d):
    def __init__(self,
            in_size: tuple[int,int,int],  # (channels, width, height)
            out_channels: int,
            kernel_size: int,
            stride: int=1):
        super().__init__(in_size[0], out_channels, kernel_size=kernel_size, stride=stride)
        self.size = (
            out_channels,
            (in_size[1] - kernel_size)//stride + 1,
            (in_size[2] - kernel_size)//stride + 1
        )

class FlattenLayer(torch.nn.Flatten):
    def __init__(self, in_size: tuple[int,int,int]):
        super().__init__(start_dim=1)
        self.size = (in_size[0]*in_size[1]*in_size[2],)

class DenseLayer(torch.nn.Linear):
    def __init__(self, in_size: tuple[int,], out_features: int):
        super().__init__(in_size[0], out_features)
        self.size = (out_features,)

class ReluLayer(torch.nn.ReLU):
    def __init__(self, in_size: tuple[int]):
        super().__init__()
        self.size = in_size
