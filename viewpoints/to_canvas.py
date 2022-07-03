import tkinter as tk
import numpy as np
import torch

from project import Project
from viewpoint import Viewpoint, get_palette

def rescale(x: np.ndarray):
    if x.max() == x.min():
        return np.zeros(x.shape)
    return (x - x.min()) / (x.max() - x.min())

def onto_canvas(project: Project, viewpoint: Viewpoint, canvas: tk.Canvas, tensor: torch.Tensor):
    layer_index_and_property = [
        (viewpoint.layer, viewpoint.x),
        (viewpoint.layer, viewpoint.y),
        (viewpoint.layer, viewpoint.color),
    ]
    tensors = project.get_layer_properties(tensor, layer_index_and_property)
    count = np.product(tensors[0].size())
    print(count)
    arrays = [
        rescale(tensors[0].rename(None).reshape(count).numpy()) * (canvas.winfo_screenwidth()-20) / 2,
        rescale(tensors[1].rename(None).reshape(count).numpy()) * (canvas.winfo_screenheight()-20) / 2,
        (get_palette(viewpoint.palette, tensors[2].numpy().reshape((count,1,1))).reshape((count,3)) * 255).astype('uint8')
    ]
    for i in range(count):
        x = arrays[0][i]
        y = arrays[1][i]
        r,g,b = arrays[2][i]
        canvas.create_rectangle(x, y, x+20, y+20, fill=f'#{r:02x}{g:02x}{b:02x}')
    