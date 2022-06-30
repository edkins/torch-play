from typing import Literal
import torch

ShapeKind = Literal['grey2d', 'flat']

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

    def tensor_size(self, kind: ShapeKind) -> tuple[str]:
        if kind == 'grey2d':
            return (self.w, self.h, self.d)
        elif kind == 'flat':
            return (self.w * self.h * self.d,)
        else:
            raise ValueError(f'Unknown kind: {kind}')

    def remap_layers(self, from_kind: ShapeKind, to_kind: ShapeKind) -> list[torch.nn.Module]:
        from_size = self.tensor_size(from_kind)
        to_size = self.tensor_size(to_kind)
        result = []
        if from_size != to_size:
            if len(from_size) != 1:
                result.append(torch.nn.Flatten(1))
            if len(to_size) != 1:
                result.append(torch.nn.UnFlatten(1, to_size))
        return result
