from __future__ import annotations
from typing import Sequence, Optional, Callable
from data import Dataset
from layer import Layer
from shape import ShapeKind
from torch import nn
import torch
import numpy as np
from visualize import to_image, ImageStuff
from shape import Shape
from tasks import TaskManager, Task

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
    return result, result_list

class ExperimentModel(nn.Module):
    def __init__(self, torch_layers: Sequence[torch.nn.Module]):
        super(ExperimentModel, self).__init__()
        self.transforms = torch.nn.ModuleList(torch_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.transforms:
            x = layer(x)
        return x

    def get_arrays(self, x: torch.Tensor) -> list[np.ndarray]:
        self.eval()
        result = []
        for layer in self.transforms:
            x = layer(x)
            result.append(x.detach().cpu().numpy())
        return result

class Records:
    def __init__(self, num_epochs: int, state_template: dict[str,torch.Tensor]):
        self.record = {name: np.zeros((num_epochs,) + state.size()) for name,state in state_template.items()}

    def store_state_dict(self, epoch: int, state_dict: dict[str,torch.Tensor]):
        for name in state_dict:
            self.record[name][epoch] = state_dict[name].detach().cpu().numpy()

    def retrieve_state_dict(self, epoch: int, device: str):
        return {name: torch.tensor(self.record[name][epoch,:]).to(device) for name in self.record}


class Experiment(Task):
    def __init__(self, layers: Sequence[Layer], dataset: Dataset, max_epochs: int, batch_size: int, device: str, task_manager: TaskManager):
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
        self.task_manager = task_manager
        self.generator = None
        self.running = False
        self.epoch = 0
        self.epoch_callback = None
        self.dataloader_train, self.dataloader_test = self.dataset.loaders(batch_size)
        self.loss_fn = nn.CrossEntropyLoss(reduction='sum')
        self.model = self.create_model()[0]
        self.optimizer = torch.optim.Adam(self.model.parameters())
        self.records = Records(max_epochs, self.model.state_dict())
        self.records.store_state_dict(self.epoch, self.model.state_dict())

    def create_model(self) -> tuple[ExperimentModel, list[int]]:
        torch_layers, indices = sequential_layers(self.layers, self.dataset.input_kind(), self.dataset.output_kind())
        return ExperimentModel(torch_layers).to(self.device), indices

    def run_yield(self):
        for i in range(self.max_epochs):
            self.epoch = i
            for _ in self.train_epoch():
                yield None
            for _ in self.test_epoch():
                yield None
            self.records.store_state_dict(self.epoch, self.model.state_dict())

    def start(self, epoch_callback: Callable):
        if self.generator == None:
            self.generator = self.run_yield()
        self.running = True
        self.epoch_callback = epoch_callback
        self.task_manager.include_task(self)

    def pause(self):
        self.running = False

    def tick(self) -> bool:
        if not self.running or self.generator == None:
            self.task_manager.remove_task(self)
            return False
        try:
            next(self.generator)
            return True
        except StopIteration:
            self.running = False
            self.task_manager.remove_task(self)
            return False

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
                if self.epoch_callback != None:
                    self.epoch_callback(self.epoch)
            yield None

    def test_epoch(self):
        size = len(self.dataloader_test.dataset)
        # Set to evaluation mode
        self.model.eval()
        correct = torch.zeros((), dtype=torch.int64).to(self.device)
        with torch.no_grad():
            for batch_num, (X, y) in enumerate(self.dataloader_test):
                X, y = X.to(self.device), y.to(self.device)
                # Compute prediction error
                pred = self.model(X)
                predicted = pred.argmax(dim=1)
                correct += (predicted == y).sum()
                yield None
        print (f"Test accuracy: {correct.item() / size:.2f}")

    def get_snapshot(self, epoch: int) -> Snapshot:
        parameters = self.records.retrieve_state_dict(epoch, self.device)
        temp_model, indices = self.create_model()
        temp_model.load_state_dict(parameters)
        return Snapshot(temp_model, indices, self.layers, self.dataset, self.batch_size, self.device)

class Snapshot:
    def __init__(self, model: ExperimentModel, indices: list[int], layers: list[Layer], dataset: Dataset, batch_size: int, device: str):
        self.model = model
        self.indices = indices
        self.layers = layers
        self.dataset = dataset
        self.batch_size = batch_size
        self.device = device

    def get_shape(self, i: int) -> Shape:
        if i == -1:
            return self.layers[0].shape_in()
        else:
            return self.layers[i].shape_out()

    def get_images(self, index: int) -> list[ImageStuff]:
        x = self.dataset.get_train_x(index).to(self.device)
        result = [to_image(x.cpu().numpy(), self.layers[0].shape_in(), self.layers[0].kind_in())]

        # extend x with zeros to have size batch_size
        x = torch.cat((
            x.reshape((1, *x.size())),
            torch.zeros((self.batch_size - 1, *x.size())).to(self.device)
        ), axis=0)

        arrays = self.model.get_arrays(x)
        result += [to_image(arrays[self.indices[i]][0], self.layers[i].shape_out(), self.layers[i].kind_out()) for i in range(len(self.layers))]
        return result

class DummySnapshot:
    def __init__(self, layers: list[Layer], dataset: Dataset):
        self.layers = layers
        self.dataset = dataset

    def get_images(self, index: int) -> list[Optional[ImageStuff]]:
        x = self.dataset.get_train_x(index)
        result = [to_image(x.cpu().numpy(), self.layers[0].shape_in(), self.layers[0].kind_in())]
        result += [None] * len(self.layers)
        return result