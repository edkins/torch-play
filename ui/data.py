from __future__ import annotations
import torch
from typing import Literal
from torchvision import datasets
from torchvision.transforms import ToTensor
from shape import Shape, ShapeKind
import numpy as np

from visualize import to_image

# name: name of the dataset
# train: The training data, a torch.utils.data.Dataset
# test: The test data, also a torch.utils.data.Dataset
# descriptions: A string describing each corresponding label value
# train_n: The size of the training set
# test_n: The size of the test set
# channels: The number of channels in the image (1 for grayscale, 3 for RGB)
# width: The width of the image
# height: The height of the image
# labels: The number of possible label values
class Dataset:
    def __init__(self, name: str, train: torch.utils.data.Dataset, test: torch.utils.data.Dataset, descriptions: list[str]):
        self.name = name
        self.train = train
        self.test = test
        self.descriptions = descriptions

        self.train_n = len(self.train)
        self.test_n = len(self.test) 
        self.channels, self.width, self.height = self.train[0][0].size()
        channels, width, height = self.test[0][0].size()
        self.labels = len(self.descriptions)

        if self.channels != channels or self.width != width or self.height != height:
            raise ValueError("Training and test must have the same image size.")
    
    # Returns a loader for the training data and the test data with the given batch size
    def loaders(self, batch_size: int) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
        return (torch.utils.data.DataLoader(self.train, batch_size=batch_size),
                torch.utils.data.DataLoader(self.test, batch_size=batch_size))

    def input_shape(self) -> Shape:
        return Shape(self.width, self.height, self.channels)

    def input_kind(self) -> ShapeKind:
        return 'grey2d'

    def output_shape(self) -> Shape:
        return Shape(1, 1, self.labels)

    def output_kind(self) -> ShapeKind:
        return 'flat'

    def get_train_x(self, index: int) -> torch.Tensor:
        return self.train[index][0]

    def get_train_y(self, index: int) -> int:
        return self.train[index][1]

    #def get_train_image(self, index: int) -> tuple[np.ndarray, Literal['L','RGB']]:
    #    return to_image(self.train[index][0].detach().to('cpu').numpy(), self.input_shape(), self.input_kind())

# Represents a collection of datasets
class Library:
    def __init__(self, datasets: list[Dataset]):
        self.datasets = datasets

    @staticmethod
    def build() -> Library:
        return Library([
            Dataset("FashionMNIST", datasets.FashionMNIST(root='data', train=True, download=True, transform=ToTensor()), datasets.FashionMNIST(root='data', train=False, download=True, transform=ToTensor()), ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]),
            Dataset("MNIST", datasets.MNIST(root='data', train=True, download=True, transform=ToTensor()), datasets.MNIST(root='data', train=False, download=True, transform=ToTensor()), ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]),
            #Dataset("CIFAR10", datasets.CIFAR10(root='data', train=True, download=True, transform=ToTensor()), datasets.CIFAR10(root='data', train=False, download=True, transform=ToTensor()), ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]),
            #Dataset("CIFAR100", datasets.CIFAR100(root='data', train=True, download=True, transform=ToTensor()), datasets.CIFAR100(root='data', train=False, download=True, transform=ToTensor()), ["apple", "aquarium fish", "baby", "bear", "beaver", "bed", "bee", "beetle", "bicycle", "bott
        ])

    # def gui_chooser(self, master: tk.Widget, variable: IntVar) -> tk.Widget:
    #     chooser = ttk.Frame(master)
    #     # Pack buttons horizontally
    #     for i, dataset in enumerate(self.datasets):
    #         tk.Radiobutton(chooser, text=dataset.name, variable=variable, value=i, indicatoron=0).pack(side=tk.LEFT)
    #     chooser.pack(side=tk.TOP)
    #     return chooser

    def options(self) -> list[str]:
        return [dataset.name for dataset in self.datasets]

    def get_dataset_with_name(self, name: str) -> Dataset:
        for dataset in self.datasets:
            if dataset.name == name:
                return dataset
        raise ValueError(f"No dataset with name {name}")