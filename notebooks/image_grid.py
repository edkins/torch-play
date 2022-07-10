from __future__ import annotations
import matplotlib as mpl
from matplotlib import pyplot as plt
import torch
from sklearn.decomposition import PCA

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
        norm = mpl.colors.CenteredNorm()
        for row in range(self.nrows):
            for col in range(self.ncols):
                plt.subplot(self.nrows, self.ncols, row*self.ncols+col+1)
                plt.xticks([])
                plt.yticks([])
                plt.imshow(norm(self.tensor[row,col].to('cpu').numpy()), norm=mpl.colors.NoNorm(), cmap=mpl.cm.coolwarm)
        plt.suptitle(title)
        plt.show()

    def _to_column_matrix(self) -> torch.Tensor:
        return self.tensor.transpose(1,3).reshape(self.nrows * self.imw * self.imh, self.ncols)

    def _from_column_matrix(self, matrix: torch.Tensor) -> ImageGrid:
        return ImageGrid(matrix.reshape(self.nrows, self.imw, self.imh, self.ncols).transpose(1,3))

    def pca_matrix_cols(self) -> torch.Tensor:
        X = self._to_column_matrix().to('cpu').numpy()
        pca = PCA(n_components=self.ncols)
        pca.fit(X)
        return torch.from_numpy(pca.components_).to(self.tensor.device).t()
    
    def mul_cols(self, matrix: torch.Tensor) -> ImageGrid:
        if len(matrix.size()) != 2:
            raise ValueError(f'ImageGrid.mult_cols must be 2D {matrix.size()}')
        if matrix.size(0) != matrix.size(1):
            raise ValueError(f'ImageGrid.mult_cols must be square {matrix.size()}')
        if matrix.size(0) != self.ncols:
            raise ValueError(f'ImageGrid.mult_cols must have same number of columns as ImageGrid.tensor {matrix.size()} {self.ncols}')
        return self._from_column_matrix(self._to_column_matrix().matmul(matrix))
