import torch
import time
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.manifold import TSNE

from train import get_dataset


NUM = 2000
pca_cls = PCA(n_components=3)
# pca_cls = LDA(n_components=3)
# pca_cls = TSNE(n_components=3)

train_loader, test_loader = get_dataset(1, 1, 6000)

train_x = torch.flatten(train_loader.dataset.data, 1).numpy()
train_y = train_loader.dataset.targets.numpy()

test_x = torch.flatten(test_loader.dataset.data, 1).numpy()[:NUM]
test_y = test_loader.dataset.targets.numpy()[:NUM]

start = time.time()
pca_cls.fit(train_x, train_y)
print(time.time() - start)
start = time.time()
new_X = pca_cls.transform(test_x)
# new_X = pca_cls.fit_transform(test_x)
new_y = test_y
print(time.time() - start)

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.scatter(new_X[:, 0], new_X[:, 1], new_X[:, 2], c=new_y)
# plt.show()