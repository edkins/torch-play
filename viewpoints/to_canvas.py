import tkinter as tk
import numpy as np
import torch

from project import Project
from viewpoint import Viewpoint, get_palette
from portfolio import CLASS_LABELS

def rescale(x: np.ndarray):
    if x.max() == x.min():
        return np.zeros(x.shape)
    return (x - x.min()) / (x.max() - x.min())

def onto_canvas(project: Project, viewpoint: Viewpoint, canvas: tk.Canvas, tensor: torch.Tensor):
    layer_index_and_property = [
        (viewpoint.layer, viewpoint.x),
        (viewpoint.layer, viewpoint.y),
        (viewpoint.layer, viewpoint.color),
        (viewpoint.layer, viewpoint.labels or '0'),
    ]
    size = viewpoint.size
    tensors = project.get_layer_properties(tensor, layer_index_and_property)
    count = np.product(tensors[0].size())
    csize = min(canvas.winfo_width(), canvas.winfo_height()) - size - 2
    arrays = [
        rescale(tensors[0].rename(None).reshape(count).numpy()) * csize,
        rescale(tensors[1].rename(None).reshape(count).numpy()) * csize,
        (get_palette(viewpoint.palette, tensors[2].numpy().reshape((count,1,1))).reshape((count,3)) * 255).astype('uint8'),
    ]

    if viewpoint.labels is not None:
        label_indices = tensors[3].numpy().reshape(count).astype('uint8')

    for i in range(count):
        x = arrays[0][i]
        y = arrays[1][i]
        r,g,b = arrays[2][i]
        if viewpoint.labels is not None:
            label = CLASS_LABELS[viewpoint.labels][label_indices[i]]
            canvas.create_text(x + size*1.5, y + size*0.5, text=label, fill='black', anchor='w')
        canvas.create_rectangle(x+1, y+1, x+size-1, y+size-1, fill=f'#{r:02x}{g:02x}{b:02x}', outline='')
    