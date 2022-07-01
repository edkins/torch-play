import numpy as np
import math
import PIL
import tkinter as tk
from tkinter import ttk

from typing import Literal, Optional
from shape import Shape, ShapeKind

ImageMode = Literal['L','RGB']
ImageStuff = tuple[np.ndarray, ImageMode]

class Artifact:
    def __init__(self, array: np.ndarray, shape: Shape, kind: ShapeKind, correct_class: Optional[int] = None):
        self.array = array
        self.shape = shape
        self.kind = kind
        self.correct_class = correct_class

    def to_image(self, width: int, height: int) -> PIL.Image:
        if self.correct_class != None:
            stuff = to_output_image(self.array, self.correct_class)
        else:
            stuff = to_image(self.array, self.shape, self.kind)
        return PIL.Image.fromarray(*stuff).resize((width, height), PIL.Image.NEAREST)

    def coords_to_neuron(self, x: int, y: int, width: int, height: int) -> Optional[tuple[int]]:
        if self.shape.d == 1:
            nx = (x * self.shape.w) // width
            ny = (y * self.shape.h) // height
            nc = 0
        elif self.shape.w == 1 and self.shape.h == 1:
            nx = 0
            ny = 0
            w = math.ceil(math.sqrt(shape.d))
            h = h
            x0 = (x * w) // width
            y0 = (y * h) // height
            nc = (x0 + y0 * w)
        else:
            raise NotImplementedError(f"Can't convert to neuron coordinates for this shape. {self.shape}")

        if self.kind == ShapeKind.flat:
            return (nc + nx * self.shape.d + ny * self.shape.d * self.shape.w),
        elif self.kind == ShapeKind.grey2d:
            if nc != 0:
                raise ValueError(f"Not expecting a neuron class for this layer: {self.shape} {nx} {ny} {nc}")
            return nx, ny
        else:
            raise ValueError(f"Unknown kind: {self.kind}")

# Clamp to [0,1], multiply by 255 and cast to uint8
def uint8ify(x: np.ndarray) -> np.ndarray:
    return (x.clip(0,1) * 255).astype('uint8')

def uint8_softclamp(x: np.ndarray) -> np.ndarray:
    return uint8ify(np.tanh((x - 0.5) * 2) * 0.45 + 0.55)

def to_image(x: np.ndarray, shape: Shape, kind: ShapeKind) -> ImageStuff:
    #print(f'to_image: {x.shape} {shape} {kind}')
    if shape.d == 1:
        return uint8_softclamp(x.reshape(shape.w, shape.h)), 'L'
    elif shape.w == 1 and shape.h == 1:
        width = math.ceil(math.sqrt(shape.d))
        height = width
        extra = np.zeros((width * height - shape.d,),dtype='uint8')
        return np.concatenate((uint8_softclamp(x), extra)).reshape((width,height)), 'L'
    else:
        raise NotImplementedError(f"Only single channel images are currently supported. shape={shape}. x.shape={x.shape}")

def to_output_image(x: np.ndarray, y: int) -> ImageStuff:
    if len(x.shape) != 1:
        raise NotImplementedError(f"Only single-dimension output vectors are supported. x.shape={x.shape}")
    length = x.shape[0]
    if y < 0 or y > length:
        raise ValueError(f"y={y} is out of range. x.shape={x.shape}")
    colours = np.array([[0,1,0] if i==y else [1,0,0] for i in range(length)])
    column = uint8_softclamp(x.reshape((length,1)) * colours)
    width = math.ceil(math.sqrt(length))
    height = width
    extra = np.zeros((width * height - length,3),dtype='uint8')
    return np.concatenate((column, extra)).reshape((width,height,3)), 'RGB'

class Visualizer:
    def __init__(self, master: tk.Misc, column: Optional[int]=None, row: Optional[int]=None, width: Optional[int]=None, height: Optional[int]=None):
        self.label = tk.Label(master)
        if column != None:
            self.label.grid(column=column, row=row)
        self.tk_image = None
        if width != None:
            self.width = width
            self.height = height
            self.label.place(width=width, height=height)
        else:
            self.width = 1
            self.height = 1

    def place(self, x: int, y: int):
        self.label.place(x=x, y=y, width=self.width, height=self.height)

    def destroy(self):
        self.label.destroy()
    
    def set(self, artifact: Optional[Artifact]):
        if artifact == None:
            self.tk_image = None
            self.label.config(image=None)
        else:
            # Store in self or else the image might be prematurely garbage collected
            self.tk_image = PIL.ImageTk.PhotoImage(artifact.to_image(self.width, self.height))
            self.label.config(image=self.tk_image)
