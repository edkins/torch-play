import numpy as np
import math
from typing import Literal
from shape import Shape, ShapeKind

ImageMode = Literal['L','RGB']
ImageStuff = tuple[np.ndarray, ImageMode]

def to_image(x: np.ndarray, shape: Shape, kind: ShapeKind) -> ImageStuff:
    #print(f'to_image: {x.shape} {shape} {kind}')
    if shape.d == 1:
        return (x.reshape(shape.w, shape.h) * 255).astype('uint8'), 'L'
    elif shape.w == 1 and shape.h == 1:
        width = math.ceil(math.sqrt(shape.d))
        height = width
        extra = np.zeros((width * height - shape.d,),dtype='uint8')
        return np.concatenate(((x * 255).astype('uint8'), extra)).reshape((width,height)), 'L'
    else:
        raise NotImplementedError(f"Only single channel images are currently supported. shape={shape}. x.shape={x.shape}")

def to_output_image(x: np.ndarray, y: int) -> ImageStuff:
    if len(x.shape) != 1:
        raise NotImplementedError(f"Only single-dimension output vectors are supported. x.shape={x.shape}")
    length = x.shape[0]
    if y < 0 or y > length:
        raise ValueError(f"y={y} is out of range. x.shape={x.shape}")
    colours = np.array([[0,255,0] if i==y else [255,0,0] for i in range(length)])
    column = (x.reshape((length,1)) * colours).astype('uint8')
    width = math.ceil(math.sqrt(length))
    height = width
    extra = np.zeros((width * height - length,3),dtype='uint8')
    return np.concatenate((column, extra)).reshape((width,height,3)), 'RGB'
