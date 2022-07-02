import torch
import PIL

class ImageViewpoint:
    def __init__(self, x: str, y: str):
        self.x = x
        self.y = y

    def to_image(self, tensor: torch.Tensor) -> PIL.Image:
        return PIL.Image.fromarray((tensor.detach().cpu().numpy().clip(0,1) * 255).astype('uint8'), 'L')
    

    

class Viewpoint:
    def __init__(self, name: str):
        self.name = name
