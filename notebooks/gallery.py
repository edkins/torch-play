from typing import Callable

from architecture import LinearSoftmax
from model import Model
from data_adapters import DataAdapter, mnist, fashion_mnist

def _mnist(data:str) -> Callable[[],DataAdapter]:
    if data == 'mnist':
        return mnist
    elif data == 'fashion':
        return fashion_mnist
    else:
        raise ValueError(f'Unknown data type: {data}')

def mnist_linear(data:str='mnist') -> Model:
    return Model(_mnist(data), lambda:LinearSoftmax(28*28, 10))
