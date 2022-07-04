import torch
import torchvision.datasets

from project import Project
from viewpoint import Viewpoint, ImageViewpoint
from layers import FlattenLinear
from my_datasets import MyFashionMNIST

CLASS_LABELS = {
    'class_fashion': torchvision.datasets.FashionMNIST.classes
}

def create_projects() -> list[Project]:
    return [
        Project(
            name='FashionMNIST Linear',
            dataset_class=MyFashionMNIST,
            in_size=(('y',28),('x',28)),
            out_size=(('class_fashion',10),),
            layers=[
                FlattenLinear((('y',28),('x',28)), (('class_fashion',10),)),
                torch.nn.Softmax(dim=1),
            ],
            viewpoints=[
                Viewpoint('Input', layer=-1, x='x', y='y', size=30, palette='white-black'),
                Viewpoint('Layer', layer=0, x='0', y='class_fashion', size=40, palette='red-blue', labels='class_fashion'),
                Viewpoint('Output', layer=1, x='0', y='class_fashion', size=40, palette='black-white', labels='class_fashion'),
            ],
            train_preview=ImageViewpoint(x='x',y='y',palette='white-black')
        ),
    ]
