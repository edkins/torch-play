import json
import os
import math
import numpy as np
from typing import Callable, Optional
from typing_extensions import Self
import torch
import matplotlib.pyplot as plt

from data_adapters import DataAdapter

def translate_device(device: str) -> torch.device:
    if device == 'default':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        return torch.device(device)

def load_parameters_json(filename: str) -> dict[str, torch.Tensor]:
    with open(filename, 'r') as f:
        json_dicts = json.load(f)
    model_state_dict = {}
    for key,value in json_dicts['model'].items():
        model_state_dict[key] = torch.tensor(value)
    return model_state_dict

def plot_tensor(img: torch.Tensor) -> None:
    if len(img.size()) == 3:
        n_figures = img.size(0)
        for i in range(n_figures):
            plt.subplot(n_figures, 1, i+1)
            plt.imshow(img[i].to('cpu').numpy())
    elif len(img.size()) == 1:
        n = img.size(0)
        w = math.ceil(math.sqrt(n))
        h = math.ceil(n/w)
        plt.subplot(1, 1, 1)
        square = np.zeros((w*h,))
        square[:n] = img.to('cpu').numpy()
        plt.imshow(square.reshape((w,h)))
    else:
        raise ValueError('img dimensions must be 1 or 3')
    plt.show()

def save_parameters_json(filename: str, model_state_dict: dict[str, torch.Tensor]) -> None:
    model_json_dict = {
        key: value.tolist() for key,value in model_state_dict.items()
    }
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w') as f:
        json.dump({'model': model_json_dict}, f)


class Model:
    def __init__(self,
            data_fn: Callable[[],DataAdapter],
            model_fn: Callable[[],torch.nn.Module],
            device: str='default',
            record: bool=True,
            batch_size=64):
        self.data_fn = data_fn
        self.train_data = None
        self.test_data = None
        self.model_fn = model_fn
        self.model = None
        self.optimizer = None
        self.device = translate_device(device)
        self.batch_size = batch_size
        self.record = record
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.num_epochs_completed = 0
        self.activations = None

    def load_data(self):
        if self.train_data is None:
            print('Loading data...')
            self.train_data = self.data_fn(train=True, device=self.device)
            self.test_data = self.data_fn(train=False, device=self.device)

    def load_model(self):
        if self.model is None:
            print('Creating model...')
            self.model = self.model_fn().to(self.device)
            self.optimizer = torch.optim.Adam(self.model.parameters())
            #self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)

    def get_filename(self, epochs: int) -> str:
        self.load_data()
        self.load_model()
        return f'parameters/model_{self.train_data}_{self.model}_{epochs}.json'

    def get_most_recent_filename(self, epochs: int) -> tuple[Optional[str],Optional[int]]:
        result = None, None
        for e in range(1, epochs + 1):
            filename = self.get_filename(e)
            if os.path.exists(filename):
                result = filename, e
        return result

    def delete_history(self) -> Self:
        filenames = []
        e = 1
        while True:
            filename = self.get_filename(e)
            if os.path.exists(filename):
                filenames.append(filename)
                e += 1
            else:
                break
        for filename in reversed(filenames):
            os.remove(filename)
            print(f"Deleted {filename}")
        return self

    def train(self, epochs: int) -> Self:
        self.load_data()
        self.load_model()
        self.activations = None
        filename, loaded_epoch = self.get_most_recent_filename(epochs)
        if filename is not None:
            model_state_dict = load_parameters_json(filename)
            self.model.load_state_dict(model_state_dict)
            self.num_epochs_completed = loaded_epoch
            print(f"Loaded model parameters from {filename}")
            print(f"Loaded up to epoch {loaded_epoch}")
        for e in range(self.num_epochs_completed + 1, epochs + 1):
            self._train_epoch()
            self.test()
            save_filename = self.get_filename(e)
            if self.record:
                save_parameters_json(save_filename, self.model.state_dict())
                print(f"Saved model parameters to {save_filename}")
            self.num_epochs_completed = e
            print(f'Completed epoch {e}')
        return self

    def _train_epoch(self) -> None:
        size = len(self.train_data)
        # Set to training mode
        self.model.train()
        for batch_num, (X, y) in enumerate(self.train_data.get_all_xy_in_batches(self.batch_size, shuffle=True)):
            # Compute prediction error
            pred = self.model(X)
            loss = self.loss_fn(pred, y)
            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if batch_num % 100 == 0:
                loss, current = loss.item(), batch_num * len(X)
                print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")

    def test(self) -> Self:
        self.load_data()
        self.load_model()
        size = len(self.test_data)
        # Set to evaluation mode
        self.model.eval()
        correct = torch.zeros((), dtype=torch.int64).to(self.device)
        with torch.no_grad():
            for batch_num, (X, y) in enumerate(self.test_data.get_all_xy_in_batches(self.batch_size, shuffle=False)):
                # Compute prediction error
                pred = self.model(X)
                predicted = pred.argmax(dim=1)
                correct += (predicted == y).sum()
        print (f"Test accuracy: {correct.item() / size:.2f}")
        return self

    def activate(self, num_samples: int) -> Self:
        self.load_data()
        self.load_model()
        self.model.eval()
        X, y = self.train_data.get_first_few_xy(num_samples)
        self.activations = self.model.forward_steps(X)
        return self

    def show_activation(self, index: int):
        if self.activations == None:
            raise Exception('Activations not computed')
        for alayer in self.activations:
            img = alayer[index]
            print(img.shape)
            plot_tensor(img)

