from torch import nn

class Shape:
    def __init__(self, w, h, d):
        self.w = w
        self.h = h
        self.d = d

    def __repr__(self):
        if self.w == 1 and self.h == 1:
            return f'{self.d}'
        elif self.d == 1:
            return f'{self.w}x{self.h}'
        else:
            return f'{self.w}x{self.h}x{self.d}'

    def count(self) -> int:
        return self.w * self.h * self.d

    def __eq__(self, other):
        return (self.w, self.h, self.d) == (other.w, other.h, other.d)

class Layer:
    def __init__(self, shape_in: Shape, shape_out: Shape, repr: str):
        self.shape_in = shape_in
        self.shape_out = shape_out
        self.repr = repr

    def __repr__(self):
        return self.repr

    def to_torch(self) -> nn.Module:
        ...

class InputLayer(Layer):
    def __init__(self, shape_in: Shape):
        super().__init__(shape_in, shape_in, f'input({shape_in})')

    # Is this right? Or should we just skip the input layer?
    def to_torch(self) -> nn.Module:
        return nn.Identity()

class DenseLayer(Layer):
    def __init__(self, shape_in: Shape, shape_out: Shape):
        super().__init__(shape_in, shape_out, f'dense({shape_out})')

    def to_torch(self) -> nn.Module:
        return nn.Linear(self.shape_in.count(), self.shape_out.count())

class SoftMaxLayer(Layer):
    def __init__(self, shape: Shape):
        super().__init__(shape, shape, 'softmax')

    def to_torch(self) -> nn.Module:
        return nn.Softmax(dim=1)

available_layer_types = ('dense', 'softmax')
default_layer_type = 'dense'
def create_layer(layer_type: str, shape_in: Shape, shape_out: Shape):
    if layer_type == 'dense':
        return DenseLayer(shape_in, shape_out)
    elif layer_type == 'softmax':
        return SoftMaxLayer(shape_out)
    else:
        raise ValueError(f'Unknown layer type: {layer_type}')
