from typing import Any, Callable, Optional, Sequence, Tuple
import torchvision
import torch
import PIL

from tasks import TaskManager, Task
from layers import NamedShape, SAMPLE
from viewpoint import Viewpoint, ImageViewpoint

class ExperimentModel(torch.nn.Module):
    def __init__(self, layers: Sequence[torch.nn.Module]):
        super().__init__()
        self.layers = torch.nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x

class Project(Task):
    def __init__(self,
            name: str, dataset_class: Callable,
            in_size: NamedShape, out_size: NamedShape,
            layers: list[torch.nn.Module],
            viewpoints: list[Viewpoint],
            max_epochs: int=10, batch_size=64,
            loss_fn=torch.nn.CrossEntropyLoss(), optimizer_fn=torch.optim.Adam,
            train_preview:ImageViewpoint = ImageViewpoint(x='x',y='y')):
        self.name = name
        self.dataset_class = dataset_class
        self.in_size = in_size
        self.out_size = out_size
        self.max_epochs = max_epochs
        self.layers = layers
        self.batch_size = batch_size
        self.loss_fn = loss_fn
        self.optimizer_fn = optimizer_fn

        # Stuff that will get initialized when project is invoked (e.g. training is started)
        self.train_data = None
        self.test_data = None
        self.model = None
        self.device = None
        self.optimizer = None

        # Training/task stuff
        self.running = False
        self.task_manager = None
        self.progress_callback = None
        self.generator = None
        self.finished = False
        self.num_epochs_completed = 0

        # Things that aren't involved in the training process but are used for visualization
        self.viewpoints = viewpoints
        self.train_preview = train_preview

    def get_viewpoint(self, name: str) -> Optional[Viewpoint]:
        for v in self.viewpoints:
            if v.name == name:
                return v
        return None

    def init_data_and_model(self) -> None:
        if self.train_data != None:
            return

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.train_data = self.dataset_class(root='./data', train=True, download=True, transform=self.to_tensor)
        self.test_data = self.dataset_class(root='./data', train=False, download=True, transform=self.to_tensor)
        self.model = self.manufacture_model(state_dict=None).to(self.device)
        self.optimizer = self.optimizer_fn(self.model.parameters())

    def to_tensor(self, x: Any) -> torch.Tensor:
        tensor = torchvision.transforms.ToTensor()(x)
        # Don't like the dummy single-element dimension for greyscale images
        if len(tensor.shape) == 3 and len(self.in_size) == 2 and tensor.shape[0] == 1:
            return tensor[0]
        return tensor

    def manufacture_model(self, state_dict:Optional[dict[str,torch.Tensor]] = None) -> ExperimentModel:
        model = ExperimentModel(self.layers)
        if state_dict != None:
            model.load_state_dict(state_dict)
        return model

    def start_training(self, task_manager: TaskManager, progress_callback: Callable) -> None:
        if self.finished:
            return
        self.init_data_and_model()
        if self.generator == None:
            self.generator = self.run_yield()
        self.running = True
        self.task_manager = task_manager
        self.progress_callback = progress_callback
        self.task_manager.include_task(self)

    def pause(self) -> None:
        self.running = False
        self.task_manager = None
    
    def tick(self) -> bool:
        if not self.running or self.generator == None:
            return False
        try:
            next(self.generator)
            return True
        except StopIteration:
            self.running = False
            self.finished = True
            return False

    def run_yield(self):
        for i in range(self.max_epochs):
            for _ in self.train_epoch():
                yield None
            for _ in self.test_epoch():
                yield None
            self.num_epochs_completed = i + 1
            #self.records.store_state_dict(self.epoch, self.model.state_dict())

    def train_epoch(self):
        size = len(self.train_data)
        # Set to training mode
        self.model.train()
        dataloader = torch.utils.data.DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True)
        for batch_num, (X, y) in enumerate(dataloader):
            X, y = X.refine_names(SAMPLE, *[n for n,_ in self.in_size]).to(self.device), y.to(self.device)
            # Compute prediction error
            pred = self.model(X).rename(None)
            loss = self.loss_fn(pred, y)
            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            #print(list(model.parameters())[0][:5])
            if batch_num % 100 == 0:
                loss, current = loss.item(), batch_num * len(X)
                print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")
                if self.progress_callback != None:
                    self.progress_callback()
            yield None

    def test_epoch(self):
        size = len(self.test_data)
        # Set to evaluation mode
        self.model.eval()
        correct = torch.zeros((), dtype=torch.int64).to(self.device)
        dataloader = torch.utils.data.DataLoader(self.test_data, batch_size=self.batch_size, shuffle=False)
        with torch.no_grad():
            for batch_num, (X, y) in enumerate(dataloader):
                X, y = X.refine_names(SAMPLE, *[n for n,_ in self.in_size]).to(self.device), y.to(self.device)
                # Compute prediction error
                pred = self.model(X).rename(None)
                predicted = pred.argmax(dim=1)
                correct += (predicted == y).sum()
                yield None
        print (f"Test accuracy: {correct.item() / size:.2f}")

    def get_training_image(self, index: int) -> PIL.Image:
        self.init_data_and_model()
        return self.train_preview.to_image(self.train_data[index][0])

    def get_test_image(self, index: int) -> PIL.Image:
        self.init_data_and_model()
        return self.train_preview.to_image(self.test_data[index][0])

    def get_training_x(self, index: int) -> torch.Tensor:
        self.init_data_and_model()
        return self.train_data[index][0].refine_names(*[n for n,_ in self.in_size])

    def get_test_x(self, index: int) -> torch.Tensor:
        self.init_data_and_model()
        return self.test_data[index][0].refine_names(*[n for n,_ in self.in_size])

    def get_all_training_y(self):
        self.init_data_and_model()
        for X,y in self.train_data:
            yield y
    
    def get_all_test_y(self):
        self.init_data_and_model()
        for X,y in self.test_data:
            yield y

    def get_tensor_property(self, x:torch.Tensor, property_name: str) -> torch.Tensor:
        if property_name == 'activation':
            return x.detach().to('cpu')
        elif property_name == '0':
            return torch.zeros_like(x).to('cpu')
        elif property_name in x.names:
            dims = [(x.shape[i] if n == property_name else 1) for i,n in enumerate(x.names)]
            r = torch.arange(x.size(property_name), dtype=torch.float).reshape(*dims).refine_names(*x.names)
            ones = torch.ones(x.size(), dtype=torch.float).refine_names(*x.names)
            return ones * r
        else:
            raise ValueError(f"Unknown tensor property: {property_name}")

    def identical_batch(self, x: torch.Tensor) -> torch.Tensor:
        names = x.names
        shape = x.shape
        x = x.rename(None).reshape((1,) + shape).expand((self.batch_size,) + shape)
        return x.refine_names(SAMPLE, *names)

    def get_layer_properties(self, x:torch.Tensor, layer_index_and_property: Sequence[Tuple[int,str]]) -> list[torch.Tensor]:
        self.init_data_and_model()
        max_layer = max(layer for layer,p in layer_index_and_property)
        results = [None] * len(layer_index_and_property)
        x = self.identical_batch(x.detach()).to(self.device)

        for i,(layer2,p) in enumerate(layer_index_and_property):
            if layer2 == -1:
                results[i] = self.get_tensor_property(x, p)

        if max_layer >= 0:
            temp_model = self.manufacture_model(self.model.state_dict()).to(self.device)
            temp_model.eval()
            for layer in range(max_layer + 1):
                x = temp_model.layers[layer](x)
                for i,(layer2,p) in enumerate(layer_index_and_property):
                    if layer2 == layer:
                        results[i] = self.get_tensor_property(x, p)

        if None in results:
            raise Exception("Something went wrong: None in result")

        return results
