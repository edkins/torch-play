import numpy as np
import math
from layer import Shape

def to_image(x: np.ndarray, shape: Shape) -> np.ndarray:
    if shape.d == 1 and len(x.shape) == 3 and x.shape[2] == 1:
        return (x[:,:,0] * 255).astype('uint8')
    elif shape.w == 1 and shape.h == 1 and len(x.shape) == 1 and x.shape[0] == shape.d:
        width = math.ceil(math.sqrt(shape.d))
        height = width
        extra = np.zeros((width * height - shape.d,),dtype='uint8')
        return np.cat(((x * 255).astype('uint8'), extra)).reshape((width,height))
    else:
        raise NotImplementedError(f"Only single channel images are currently supported. shape={shape}. x.shape={x.shape}")
