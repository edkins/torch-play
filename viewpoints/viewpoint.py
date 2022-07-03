from typing import Optional
import numpy as np
import torch
import PIL

def to_3d(t: torch.tensor) -> torch.tensor:
    size = t.size()
    return t.reshape(size + (1,) * (3 - len(size)))

def get_palette(name: str, input: np.ndarray) -> np.ndarray:
    if name == 'white-black':
        return (1 - input) * np.ones((1,1,3))
    elif name == 'black-white':
        return input * np.ones((1,1,3))
    elif name == 'red-blue':
        input = np.tanh(input)
        return np.ones((1,1,3)) + (input < 0) * input * np.array([[[0,1,1]]]) - (input >= 0) * input * np.array([[[1,1,0]]])
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
    def __init__(self, name: str, layer:int, x: str, y: str, size:int=20, color:str = 'activation', palette: str = 'black-white', labels:Optional[str] = None):
        self.name = name
        self.layer = layer
        self.size = size
        self.x = x
        self.y = y
        self.color = color
        self.palette = palette
        self.labels = labels
