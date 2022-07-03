import numpy as np
import torch
import PIL
import tkinter as tk

def to_3d(t: torch.tensor) -> torch.tensor:
    size = t.size()
    return t.reshape(size + (1,) * (3 - len(size)))

def get_palette(name: str, input: np.ndarray) -> np.ndarray:
    if name == 'white-black':
        return (1 - input) * np.ones((1,1,3))
    elif name == 'black-white':
        return input * np.ones((1,1,3))
    else:
        raise ValueError(f'Unknown palette: {name}')

class ImageViewpoint:
    def __init__(self, x: str, y: str, palette: str = 'black-white'):
        self.x = x
        self.y = y
        self.palette = palette

    def to_image(self, tensor: torch.Tensor) -> PIL.Image:
        array = get_palette(self.palette, to_3d(tensor).detach().cpu().numpy()).clip(0,1)
        return PIL.Image.fromarray((array * 255).astype('uint8'), 'RGB')
    

    

class Viewpoint:
    def __init__(self, name: str, layer:int, x: str, y: str, color:str = 'activation', palette: str = 'black-white'):
        self.name = name
        self.layer = layer
        self.x = x
        self.y = y
        self.color = color
        self.palette = palette
