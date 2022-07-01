from torch import nn
import numpy as np

from shape import ShapeKind, Shape

class Layer:
    def to_torch(self) -> nn.Module:
        ...

    def kind_in(self) -> ShapeKind:
        ...

    def kind_out(self) -> ShapeKind:
        ...


class DenseLayer(Layer):
    def __init__(self, shape_in: Shape, shape_out: Shape):
        self.s_in = shape_in
        self.s_out = shape_out

    def __repr__(self):
        return f'dense({self.s_out})'

    def to_torch(self) -> nn.Module:
        return nn.Linear(self.s_in.count(), self.s_out.count())

    def shape_in(self) -> Shape:
        return self.s_in

    def shape_out(self) -> Shape:
        return self.s_out

    def kind_in(self) -> ShapeKind:
        return 'flat'

    def kind_out(self) -> ShapeKind:
        return 'flat'

    def torch_reshape_weights(self, weights: np.ndarray, neuron: tuple[int]) -> np.ndarray:
        if len(neuron) != 1:
            raise ValueError(f'DenseLayer.torch_shape_weights: neuron must be a tuple of length 1, got {neuron}')
        return weights.reshape(self.s_out.count(), self.s_in.count())[neuron]
    

class SoftMaxLayer(Layer):
    def __init__(self, shape: Shape):
        self.shape = shape

    def __repr__(self):
        return f'softmax({self.shape})'

    def to_torch(self) -> nn.Module:
        return nn.Softmax(dim=1)

    def shape_in(self) -> Shape:
        return self.shape

    def shape_out(self) -> Shape:
        return self.shape

    def kind_in(self) -> ShapeKind:
        return 'flat'

    def kind_out(self) -> ShapeKind:
        return 'flat'


available_layer_types = ('dense', 'softmax')
default_layer_type = 'dense'
def create_layer(layer_type: str, shape_in: Shape, shape_out: Shape):
    if layer_type == 'dense':
        return DenseLayer(shape_in, shape_out)
    elif layer_type == 'softmax':
        return SoftMaxLayer(shape_out)
    else:
        raise ValueError(f'Unknown layer type: {layer_type}')
