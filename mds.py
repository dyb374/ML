# @Author   : Andrew Dong
# @time     : 2018/10/6 15:39
# @file     : mds.py
# @Software : PyCharm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import NullFormatter
from sklearn import manifold, datasets, metrics
import numpy as np


def calculate_distance(x, y):
    d = np.sqrt(np.sum((x - y) ** 2))
    return d


def calculate_distance_matrix(x, y):
    d = metrics.pairwise_distances(x, y)
    return d


def calculate_B(D):
    (n1, n2) = D.shape
    DD = np.square(D)
    Di = np.sum(DD, axis=1) / n1
    Dj = np.sum(DD, axis=0) / n1
    Dij = np.sum(DD) / (n1 ** 2)
    B = np.zeros((n1, n1))
    for i in range(n1):
        for j in range(n2):
            B[i, j] = (Dij + DD[i, j] - Di[i] - Dj[j]) / (-2)
    return B


def MDS(data, n=2):
    D = calculate_distance_matrix(data, data)
    B = calculate_B(D)
    B1, B2 = np.linalg.eigh(B)
    B1_sort = np.argsort(-B1)
    B1 = B1[B1_sort]
    B2 = B2[:, B1_sort]
    B1z = np.diag(B1[0:n])
    B2z = B2[:, 0:n]
    Y = np.dot(np.sqrt(B1z), B2z.T).T
    return Y


n_points = 1000
X, color = datasets.make_swiss_roll(n_samples=1000)
n_neighbors = 10
n_components = 2


fig = plt.figure(figsize=(15, 8))

ax = fig.add_subplot(251, projection='3d')
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=color, cmap=plt.cm.Spectral)
ax.view_init(4, -72)


Y = manifold.MDS(n_components, max_iter=100, n_init=1).fit_transform(X)
ax = fig.add_subplot(142)
plt.scatter(Y[:, 0], Y[:, 1], c=color, cmap=plt.cm.Spectral)
plt.title("sklearnMDS result")
ax.xaxis.set_major_formatter(NullFormatter())
ax.yaxis.set_major_formatter(NullFormatter())

Y = MDS(X)
ax = fig.add_subplot(143)
plt.scatter(Y[:, 0], Y[:, 1], c=color, cmap=plt.cm.Spectral)
plt.title("myMDS result")
ax.xaxis.set_major_formatter(NullFormatter())
ax.yaxis.set_major_formatter(NullFormatter())

plt.show()
