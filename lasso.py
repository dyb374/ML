# @Author   : Andrew Dong
# @time     : 2018/12/16 18:25
# @file     : lasso.py
# @Software : PyCharm
import numpy as np
import itertools
from sklearn.datasets.samples_generator import make_regression
from sklearn import preprocessing


def lasso_regression(X, y, lambd=0.3, threshold=0.1):
    # 计算残差平方和
    rss = lambda X, y, w: (y - X * w).T * (y - X * w)

    # 初始化回归系数w.
    m, n = X.shape
    w = np.matrix(np.zeros((n, 1)))
    r = rss(X, y, w)

    # 使用坐标下降法优化回归系数w
    niter = itertools.count(1)
    for i in niter:
        for k in range(n):
            # 计算常量值z_k和p_k
            z_k = (X[:, k].T * X[:, k])
            p_k = 0
            for i in range(m):
                p_k += X[i, k] * (y[i, 0] - sum([X[i, j] * w[j, 0] for j in range(n) if j != k]))
            if p_k < -lambd / 2:
                w_k = (p_k + lambd / 2) / z_k
            elif p_k > lambd / 2:
                w_k = (p_k - lambd / 2) / z_k
            else:
                w_k = 0
            w[k, 0] = w_k
        r_prime = rss(X, y, w)
        delta = abs(r_prime - r)[0, 0]
        r = r_prime
        if delta < threshold:
            break
    return w


X, y = make_regression(n_samples=500, n_features=600, n_informative=5, noise=5)


#利用sklearn的scale标准化X与y
X = preprocessing.scale(X)
y = preprocessing.scale(y)
w = lasso_regression(X, y, lambd=10)
print('Regression coefficients: ', w)
