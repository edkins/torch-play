from __future__ import annotations
from typing import Sequence
from data import Dataset
from layer import Layer
from shape import ShapeKind
from torch import nn
import torch
import numpy as np
import threading
from visualize import to_image

def sequential_layers(layers: Sequence[Layer], input_kind: ShapeKind, output_kind: ShapeKind) -> tuple[nn.Module,list[int]]:
    result = []
    result_kind = input_kind
    result_list = []
    for layer in layers:
        for module in layer.shape_in().remap_layers(result_kind, layer.kind_in()):
            result.append(module)
        result_list.append(len(result))
        result.append(layer.to_torch())
        result_kind = layer.kind_out()
    for module in layers[-1].shape_out().remap_layers(result_kind, output_kind):
        result.append(module)
    result_list.append(len(result))
    print(result)
    return result, result_list

class ExperimentModel(nn.Module):
    def __init__(self, torch_layers: Sequence[torch.nn.Module], indices: list[int]):
        super(ExperimentModel, self).__init__()
        self.transforms = torch.nn.ModuleList(torch_layers)
        self.indices = indices

    def sub_model(self, index: int) -> ExperimentModel:
        return ExperimentModel(self.transforms[:self.indices[index]], self.indices[:index])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.transforms:
            x = layer(x)
        return x

class Records:
    def __init__(self, num_epochs: int, state_template: dict[str,torch.Tensor]):
        self.record = {name: np.zeros((num_epochs,) + state.size()) for name,state in state_template.items()}

    def store_state_dict(self, epoch: int, state_dict: dict[str,torch.Tensor]):
        for name in state_dict:
            self.record[name][epoch] = state_dict[name].detach().cpu().numpy()

    def retrieve_state_dict(self, epoch: int, device: str):
        return {name: torch.tensor(self.record[name][epoch,:]).to(device) for name in self.record}

class ThreadStarter:
    def __init__(self, generator_function: callable):
        self.generator_function = generator_function
        self.thread = None
        self.generator = None
        self.want_progress = False
        self.completed = False

    def start(self):
        if self.want_progress:
            raise Exception("Thread already started")
        if self.generator == None:
            self.generator = self.generator_function()
            self.completed = False
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.want_progress = True
        self.thread.start()

    def pause(self):
        if not self.want_progress:
            raise Exception("Thread not started")
        self.want_progress = False
        self.thread.join()
        self.thread = None

    def run(self):
        try:
            while self.want_progress:
                self.generator.__next__()
        except StopIteration:
            self.completed = True

class Experiment:
    def __init__(self, layers: Sequence[Layer], dataset: Dataset, max_epochs: int, batch_size: int, device: str):
        if layers[0].shape_in() != dataset.input_shape():
            raise ValueError(f"First layer input shape {layers[0].shape_in()} does not match dataset input shape {dataset.input_shape()}")
        for i in range(len(layers)-1):
            if layers[i].shape_out() != layers[i+1].shape_in():
                raise ValueError(f"Layer {i} output shape {layers[i].shape_out()} does not match input shape {layers[i+1].shape_in}")
        if layers[-1].shape_out() != dataset.output_shape():
            raise ValueError(f"Last layer output shape {layers[-1].shape_out()} does not match dataset shape {dataset.output_shape()}")

        self.layers = layers
        self.dataset = dataset
        self.device = device
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        torch_layers, indices = sequential_layers(layers, dataset.input_kind(), dataset.output_kind())
        self.model = ExperimentModel(torch_layers, indices).to(device)
        self.records = Records(max_epochs, self.model.state_dict())
        self.epoch = 0
        self.records.store_state_dict(self.epoch, self.model.state_dict())
        self.dataloader_train, self.dataloader_test = self.dataset.loaders(batch_size)
        self.loss_fn = nn.CrossEntropyLoss(reduction='sum')
        self.optimizer = torch.optim.Adam(self.model.parameters())
        self.thread_starter = ThreadStarter(self.run_yield)

    def run_yield(self):
        for i in range(self.max_epochs):
            self.epoch = i
            for _ in self.train_epoch():
                yield None
            self.records.store_state_dict(self.epoch, self.model.state_dict())

    def start(self):
        self.thread_starter.start()

    def pause(self):
        self.thread_starter.pause()

    def progress(self) -> tuple[int, bool]:
        return self.epoch, self.thread_starter.completed

    def train_epoch(self):
        size = len(self.dataloader_train.dataset)
        # Set to training mode
        self.model.train()
        for batch_num, (X, y) in enumerate(self.dataloader_train):
            X, y = X.to(self.device), y.to(self.device)
            # Compute prediction error
            pred = self.model(X)
            loss = self.loss_fn(pred, y)
            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            #print(list(model.parameters())[0][:5])
            if batch_num % 100 == 0:
                loss, current = loss.item(), batch_num * len(X)
                print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")
            yield None

    def get_image(self, epoch: int, index: int, layer: int) -> np.ndarray:
        parameters = self.records.retrieve_state_dict(epoch, self.model.device)
        temp_model = self.model.sub_model(layer).to(self.model.device)
        temp_model.load_state_dict(parameters)
        batch_num = index // self.batch_size
        X = self.dataloader_train(batch_num)[0]
        tensor = temp_model(X)[index % self.batch_size]
        array = tensor.detach().cpu().numpy()
        return to_image(array, self.layers[layer].shape_out)
