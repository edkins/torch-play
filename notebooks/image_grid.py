from __future__ import annotations
import matplotlib as mpl
from matplotlib import pyplot as plt
import torch
from sklearn.decomposition import PCA, NMF

_positive_colors = [
    (1, 1, 1),    # white
    (1, 0, 0),    # red
    (0, 0, 0),    # black
]

_negative_colors = [
    (1, 1, 1),    # white
    (0, 0, 1),    # blue
    (0.1, 0.1, 0.1) # grey
]

_maximum = (4 ** (len(_positive_colors)-1)) / 2

_mappings = {
    color: [(0.5 - (4**(i-1))/_maximum, _negative_colors[i][ch], _negative_colors[i][ch]) for i in range(len(_negative_colors)-1,0,-1)] + [(0.5 if i==0 else 0.5 + (2**i)/_maximum, _positive_colors[i][ch], _positive_colors[i][ch]) for i in range(len(_positive_colors))]
        for ch,color in enumerate(['red','green','blue'])
}

print(_mappings)

cmap = mpl.colors.LinearSegmentedColormap("my_colormap", _mappings)

class ImageGrid:
    def __init__(self, tensor: torch.Tensor):
        if len(tensor.size()) != 4:
            raise ValueError('ImageGrid.tensor must be 4D')
        self.tensor = tensor

    def transpose(self) -> ImageGrid:
        return ImageGrid(self.tensor.transpose(0,1))

    @property
    def nrows(self) -> int:
        return self.tensor.size(0)

    @property
    def ncols(self) -> int:
        return self.tensor.size(1)

    @property
    def imw(self) -> int:
        return self.tensor.size(2)

    @property
    def imh(self) -> int:
        return self.tensor.size(3)

    @property
    def im_pixel_count(self) -> int:
        return self.tensor.size(2) * self.tensor.size(3)

    def plot(self, figsize:tuple[int,int]=(6,6), title: str='') -> None:
        mpl.rcParams['figure.figsize'] = figsize
        # if norm == 'all':
        #     data = self.tensor.to('cpu').numpy()
        #     normed_data = mpl.colors.SymLogNorm(1, 2)(data.reshape(-1)).reshape(data.shape)
        # elif norm == 'cols':
        #     normed_data = self.tensor.to('cpu').numpy()
        #     shape = (normed_data.shape[0],) + normed_data.shape[2:]
        #     for c in range(self.ncols):
        #         normed_data[:,c] = mpl.colors.SymLogNorm(1, 2)(normed_data[:,c].reshape(-1)).reshape(shape)
        # else:
        #     raise ValueError(f'ImageGrid.plot norm must be "all" or "cols" not {norm}')
        for row in range(self.nrows):
            for col in range(self.ncols):
                plt.subplot(self.nrows, self.ncols, row*self.ncols+col+1)
                plt.xticks([])
                plt.yticks([])
                plt.imshow(0.5 + self.tensor[row,col].to('cpu').numpy() / 8, norm=mpl.colors.NoNorm(), cmap=cmap)
        plt.suptitle(title)
        #plt.colorbar()
        plt.show()

    def _to_column_matrix(self) -> torch.Tensor:
        return self.tensor.transpose(1,3).reshape(self.nrows * self.imw * self.imh, self.ncols)

    def pca_matrix_cols(self) -> torch.Tensor:
        X = self._to_column_matrix().to('cpu').numpy()
        pca = PCA(n_components=self.ncols)
        pca.fit(X)
        return torch.from_numpy(pca.components_).to(self.tensor.device).t()
    
    def nmf_matrix_cols(self, n_components: int) -> torch.Tensor:
        X = self._to_column_matrix().to('cpu').numpy()
        nmf = NMF(n_components=n_components)
        nmf.fit(X)
        return torch.from_numpy(nmf.components_).to(self.tensor.device).t()
    
    def mul_cols(self, matrix: torch.Tensor) -> ImageGrid:
        if len(matrix.size()) != 2:
            raise ValueError(f'ImageGrid.mult_cols must be 2D {matrix.size()}')
        if matrix.size(0) != self.ncols:
            raise ValueError(f'ImageGrid.mult_cols must have same number of columns as ImageGrid.tensor {matrix.size()} {self.ncols}')
        matrix2 = self._to_column_matrix().matmul(matrix)
        return ImageGrid(matrix2.reshape(self.nrows, self.imw, self.imh, matrix.size(1)).transpose(1,3))
