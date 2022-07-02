import torch

NamedShape = tuple[tuple[str, int]]
SAMPLE = 'sample'

def shape_count(shape: NamedShape) -> int:
    p = 1
    for _,s in shape:
        p *= s
    return p

def shape_anon(shape: NamedShape) -> tuple[int]:
    return tuple([s for _,s in shape])

class FlattenLinear(torch.nn.Module):
    def __init__(self, in_shape: NamedShape, out_shape: NamedShape):
        super().__init__()
        self.linear = torch.nn.Linear(in_features=shape_count(in_shape), out_features=shape_count(out_shape))
        self.flatten = torch.nn.Flatten(start_dim=1)
        self.unflatten = torch.nn.Unflatten(dim=1, unflattened_size=shape_anon(out_shape))
        self.out_names = (SAMPLE,) + tuple(n for n,_ in out_shape)
        #self.unflatten = torch.nn.Unflatten(dim=out_shape[0][0], unflattened_size=out_shape)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.unflatten(self.linear(self.flatten(x.rename(None)))).refine_names(*self.out_names)
        #return self.unflatten(self.linear(self.flatten(x.rename(None))))
