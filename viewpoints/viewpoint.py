import torch
import PIL

class ImageViewpoint:
    def __init__(self, x: str, y: str, invert: bool=False):
        self.x = x
        self.y = y
        self.invert = invert

    def to_image(self, tensor: torch.Tensor) -> PIL.Image:
        array = tensor.detach().cpu().numpy().clip(0,1)
        if self.invert:
            array = 1 - array
        return PIL.Image.fromarray((array * 255).astype('uint8'), 'L')
    

    

class Viewpoint:
    def __init__(self, name: str):
        self.name = name
