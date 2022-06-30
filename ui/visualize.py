import numpy as np
import math
from typing import Literal
from shape import Shape, ShapeKind

def to_image(x: np.ndarray, shape: Shape, kind: ShapeKind) -> tuple[np.ndarray, Literal['L','RGB']]:
    if shape.d == 1:
        return (x.reshape(shape.w, shape.h) * 255).astype('uint8'), 'L'
    elif shape.w == 1 and shape.h == 1:
        width = math.ceil(math.sqrt(shape.d))
        height = width
        extra = np.zeros((width * height - shape.d,),dtype='uint8')
        return np.cat(((x * 255).astype('uint8'), extra)).reshape((width,height)), 'L'
    else:
        raise NotImplementedError(f"Only single channel images are currently supported. shape={shape}. x.shape={x.shape}")
