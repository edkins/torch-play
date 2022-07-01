import numpy as np
import math
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk
from typing import Optional

from shape import Shape, ShapeKind

class Artifact:
    def __init__(self, array: np.ndarray, shape: Shape, kind: ShapeKind, correct_class: Optional[int] = None):
        self.array = array
        self.shape = shape
        self.kind = kind
        self.correct_class = correct_class

    def to_image(self, width: int, height: int) -> Image:
        if self.correct_class != None:
            highlight=self.correct_class,
            highlight_color=[0,1,0]
            non_highlight_color=[1,0,0]
        else:
            highlight=None
            highlight_color=None
            non_highlight_color=[1,1,1]
        img = to_image(self.array, self.shape, self.kind, highlight, highlight_color, non_highlight_color)
        return img.resize((width, height), Image.NEAREST)

    def coords_to_neuron(self, x: int, y: int, width: int, height: int) -> Optional[tuple[int]]:
        if self.shape.d == 1:
            nx = (x * self.shape.w) // width
            ny = (y * self.shape.h) // height
            nc = 0
        elif self.shape.w == 1 and self.shape.h == 1:
            nx = 0
            ny = 0
            w = math.ceil(math.sqrt(self.shape.d))
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

def to_image(x: np.ndarray, shape: Shape, kind: ShapeKind, highlight: Optional[tuple[int]]=None, highlight_color: Optional[tuple[float,float,float]]=None, non_highlight_color: tuple[float,float,float]=[1,1,1]) -> Image:
    #print(f'to_image: {x.shape} {shape} {kind}')

    ones = (1,) * len(x.shape)
    color_vec = np.ones(x.shape + (1,)) * np.array(non_highlight_color).reshape(ones + (3,))
    if highlight != None:
        if len(highlight) != len(x.shape):
            raise ValueError(f"highlight.shape={highlight} is not the same length as x.shape={x.shape}")

        vec = color_vec
        for coord in highlight:
            vec = vec[coord]
        vec[:] = highlight_color

    if shape.d == 1:
        return Image.fromarray(uint8_softclamp((x.reshape(x.shape + (1,)) * color_vec).reshape(shape.w, shape.h, 3)), 'RGB')
    elif shape.w == 1 and shape.h == 1:
        width = math.ceil(math.sqrt(shape.d))
        height = width
        extra = np.zeros((width * height - shape.d,3),dtype='uint8')
        return Image.fromarray(np.concatenate((uint8_softclamp(x.reshape(x.shape + (1,)) * color_vec), extra)).reshape((width,height,3)), 'RGB')
    else:
        raise NotImplementedError(f"Only single channel images are currently supported. shape={shape}. x.shape={x.shape}")

class Visualizer:
    def __init__(self, master: tk.Misc, column: Optional[int]=None, row: Optional[int]=None, width: Optional[int]=None, height: Optional[int]=None):
        self.label = ttk.Label(master)
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
            self.tk_image = ImageTk.PhotoImage(artifact.to_image(self.width, self.height))
            self.label.config(image=self.tk_image)
