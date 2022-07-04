import torch
import torchvision.datasets
import random

class MyMNIST(torchvision.datasets.MNIST):
    def __init__(self, root: str, train: bool, device: str):
        super().__init__(root=root, train=train, download=True)
        self.data_gpu = self.data.to(device)
    
    def __getitem__(self, index):
        x = self.data_gpu[index].float() / 255.0
        y = self.targets[index].item()
        return x, y

    def get_all_y(self) -> torch.Tensor:
        return self.targets

    def get_all_xy_in_batches(self, batch_size: int, shuffle: bool):
        if shuffle:
            choices = random.choices(range(len(self.data)), k=len(self.data))
            for i in range(0, len(self.data), batch_size):
                batch_indices = choices[i:i+batch_size]
                x = self.data_gpu[batch_indices].float() / 255.0
                y = self.targets[batch_indices].long()
                yield x, y
        else:
            for i in range(0, len(self.data), batch_size):
                x = self.data_gpu[i:i+batch_size].float() / 255.0
                y = self.targets[i:i+batch_size].long()
                yield x, y

    def get_big_batch_for_tsne(self) -> torch.Tensor:
        return self.data_gpu[:1024].float() / 255.0

class MyFashionMNIST(torchvision.datasets.FashionMNIST):
    def __init__(self, root: str, train: bool, device: str):
        super().__init__(root=root, train=train, download=True)
        self.data_gpu = self.data.to(device)
    
    def __getitem__(self, index):
        x = self.data_gpu[index].float() / 255.0
        y = self.targets[index].item()
        return x, y

    def get_all_y(self) -> torch.Tensor:
        return self.targets

    def get_all_xy_in_batches(self, batch_size: int, shuffle: bool):
        if shuffle:
            choices = random.choices(range(len(self.data)), k=len(self.data))
            for i in range(0, len(self.data), batch_size):
                batch_indices = choices[i:i+batch_size]
                x = self.data_gpu[batch_indices].float() / 255.0
                y = self.targets[batch_indices].long()
                yield x, y
        else:
            for i in range(0, len(self.data), batch_size):
                x = self.data_gpu[i:i+batch_size].float() / 255.0
                y = self.targets[i:i+batch_size].long()
                yield x, y

    def get_big_batch_for_tsne(self) -> torch.Tensor:
        return self.data_gpu[:1024].float() / 255.0
