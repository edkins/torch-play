import torch
import torchvision.datasets

from project import Project
from viewpoint import Viewpoint, ImageViewpoint
from layers import FlattenLinear

CLASS_LABELS = {
    'class_fashion': torchvision.datasets.FashionMNIST.classes
}

def create_projects() -> list[Project]:
    return [
        Project(
            name='FashionMNIST Linear',
            dataset_class=torchvision.datasets.FashionMNIST,
            in_size=(('x',28),('y',28)),
            out_size=(('class_fashion',10),),
            layers=[
                FlattenLinear((('x',28),('y',28)), (('class_fashion',10),)),
                torch.nn.Softmax(dim=1),
            ],
            viewpoints=[
                Viewpoint('Input'),
            ],
            train_preview=ImageViewpoint(x='x',y='y',invert=True),
        ),
    ]
