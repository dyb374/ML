# @Author   : Andrew Dong
# @time     : 2018/10/31 12:30
# @file     : Laplacian_Eigenmaps.py
# @Software : PyCharm
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import NullFormatter
from sklearn import datasets

'''X为数据, k为邻居数量, t为权重W相关系数, n_components为返回的纬度'''
def Laplacian_Eigenmaps(X, k, t, n_components):
    m, n = np.shape(X)
    W = np.mat(np.zeros([m, m]))
    D = np.mat(np.zeros([m, m]))

    for i in range(m):
        k_index = knn(X[i, :], X, k)  # 利用knn计算邻居

        '''计算权重，构造W与D'''
        for j in range(k):
            sqDiffVector = X[i, :] - X[k_index[j], :]
            sqDiffVector = np.array(sqDiffVector) ** 2
            sqDistances = sqDiffVector.sum()
            W[i, k_index[j]] = np.math.exp(-sqDistances / t)
            D[i, i] += W[i, k_index[j]]

    '''计算拉普拉斯特征值与特征向量'''
    L = D - W
    X = np.dot(D.I, L)
    lamda, Y = np.linalg.eig(X)

    '''求最小的n_components个非零特征值对应的特征向量并返回'''
    fm, fn = np.shape(Y)
    lamdaIndicies = np.argsort(lamda)
    for i in range(fm):
        if lamda[lamdaIndicies[i]].real > 1e-5:
            n1 = lamdaIndicies[i]
            n2 = lamdaIndicies[i + 1]
            break
    print("n1 = ", n1, ", n2 = ", n2)
    return Y, n1, n2


def knn(i, X, k):
    dataSetSize = X.shape[0]
    diffMat = np.tile(i, (dataSetSize, 1)) - X
    distances = (np.array(diffMat) ** 2).sum(axis=1) ** 0.5  # 计算欧式距离
    kNeighbors = np.argsort(distances)  # 对结果排序，这边返回的是从小到大的索引值
    return kNeighbors[0:k]  # 返回前k个最近的邻居(这里包括自己)


X, color = datasets.make_swiss_roll(n_samples=2000)  # 使用瑞士卷数据
n_neighbors = 10
n_components = 2

fig = plt.figure(figsize=(15, 8))

ax = fig.add_subplot(251, projection='3d')
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=color, cmap=plt.cm.Spectral)
ax.view_init(4, -72)


Y, n1, n2 = Laplacian_Eigenmaps(X, n_neighbors, 5, n_components)
ax = fig.add_subplot(122)
Y = np.array(Y)
plt.scatter(Y[:, n1], Y[:, n2], c=color, cmap=plt.cm.Spectral)
plt.title("result")
plt.show()
