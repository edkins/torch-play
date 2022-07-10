import random
import torch
import torchvision.datasets

class DataAdapter:
    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        ...

    def __len__(self) -> int:
        ...

    def get_all_xy_in_batches(self, batch_size: int, shuffle: bool):
        ...

class MNISTAdapter:
    def __init__(self, dataset: torchvision.datasets.MNIST, device: torch.device):
        self.data_gpu = dataset.data.to(device)
        self.targets_gpu = dataset.targets.to(device)
        self.name = dataset.__class__.__name__
        self.classes = dataset.classes
    
    # def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
    #     x = self.data_gpu[index].reshape((1,28,28)).float() / 255.0
    #     y = self.targets_gpu[index]
    #     return x, y

    def __len__(self) -> int:
        return len(self.data_gpu)

    def get_all_xy_in_batches(self, batch_size: int, shuffle: bool):
        if shuffle:
            choices = random.choices(range(len(self.data_gpu)), k=len(self.data_gpu))
            for i in range(0, len(self.data_gpu), batch_size):
                batch_indices = choices[i:i+batch_size]
                x = self.data_gpu[batch_indices]
                x = x.reshape((len(x),1,28,28)).float() / 255.0
                y = self.targets_gpu[batch_indices]
                yield x, y
        else:
            for i in range(0, len(self.data_gpu), batch_size):
                x = self.data_gpu[i:i+batch_size]
                x = x.reshape((len(x),1,28,28)).float() / 255.0
                y = self.targets_gpu[i:i+batch_size]
                yield x, y
    
    def get_first_few_xy(self, n: int | str) -> tuple[torch.Tensor, torch.Tensor]:
        if isinstance(n, int):
            x = self.data_gpu[:n]
            x = x.reshape((len(x),1,28,28)).float() / 255.0
            y = self.targets_gpu[:n]
            return x, y
        elif n == 'classes':
            indices = [None] * len(self.classes)
            remaining = len(self.classes)
            for index, y in enumerate(self.targets_gpu):
                if indices[y] == None:
                    indices[y] = index
                    remaining -= 1
                    if remaining == 0:
                        break
            x = self.data_gpu[indices]
            x = x.reshape((len(x),1,28,28)).float() / 255.0
            y = self.targets_gpu[indices]
            return x, y

    def __str__(self):
        return self.name

def fashion_mnist(train: bool, device: torch.device) -> DataAdapter:
    return MNISTAdapter(torchvision.datasets.FashionMNIST(root='./data', train=train, download=True), device)

def mnist(train: bool, device: torch.device) -> DataAdapter:
    return MNISTAdapter(torchvision.datasets.MNIST(root='./data', train=train, download=True), device)
