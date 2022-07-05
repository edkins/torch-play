import torchvision.datasets
from torchvision.transforms import ToTensor
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
import numpy as np

dataset = torchvision.datasets.FashionMNIST('./data', train=True, download=True)
n = 1000 #len(dataset)
X = np.zeros((n, 28*28))
for i in range(n):
    img = dataset[i][0]
    X[i,:] = np.asarray(img, dtype='float').reshape((28*28,))

print(X.shape)

X_reduced = PCA(n_components=50).fit_transform(X)

print(X_reduced.shape)

X_transformed = TSNE(n_components=2, init='pca', verbose=2, perplexity=5).fit_transform(X_reduced)

for i, label in enumerate(dataset.classes):
    print(i, label)

print(X_transformed.shape)
plt.scatter(x=X_transformed[:,0], y=X_transformed[:,1], c=dataset.targets[:n], cmap='Paired')
plt.colorbar()
plt.show()
