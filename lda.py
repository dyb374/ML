# @Author   : Andrew Dong
# @time     : 2018/10/14 23:06
# @file     : lda.py
# @Software : PyCharm


import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_blobs


def LDA(X, y):
    X1 = np.array([X[i] for i in range(len(X)) if y[i] == 0])
    X2 = np.array([X[i] for i in range(len(X)) if y[i] == 1])

    len1 = len(X1)
    len2 = len(X2)

    '''求中心点'''
    u1 = np.mean(X1, axis=0)
    u2 = np.mean(X2, axis=0)

    '''计算Sw'''
    cov1 = np.dot((X1 - u1).T, (X1 - u1))
    cov2 = np.dot((X2 - u2).T, (X2 - u2))
    Sw = cov1 + cov2

    '''求w'''
    w = np.dot(np.mat(Sw).I, (u1 - u2).reshape((len(u1), 1)))
    print(w[0]*w[0] + w[1]*w[1])
    return w


n = 200
X, y = make_blobs(n_samples=n, centers=2, n_features=2, random_state=0)
w = LDA(X, y)

plt.scatter(X[:, 0], X[:, 1], marker='o', c=y)
plt.plot([-n * float(w[0]), n * float(w[0])], [-n * float(w[1]), n * float(w[1])], 'r', lw=2)
plt.show()
