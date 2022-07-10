from typing import Callable

from architecture import LinearSoftmax, CNN1, CNN2
from model import Model
from data_adapters import DataAdapter, mnist, fashion_mnist

def _mnist(data:str) -> Callable[[],DataAdapter]:
    if data == 'mnist':
        return mnist
    elif data == 'fashion':
        return fashion_mnist
    else:
        raise ValueError(f'Unknown data type: {data}')

def mnist_linear(data:str='mnist', **kwargs) -> Model:
    return Model(_mnist(data), lambda:LinearSoftmax(28, 28, 1, 10), **kwargs)

def mnist_cnn(data:str='mnist', attempt:int=1, **kwargs) -> Model:
    if attempt == 1:
        return Model(_mnist(data), lambda:CNN1(28, 28, 1, 10), **kwargs)
    elif attempt == 2:
        return Model(_mnist(data), lambda:CNN2(28, 28, 1, 10), **kwargs)
    else:
        raise ValueError(f'Unknown attempt: {attempt}')
