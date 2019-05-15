# @Author   : Andrew Dong
# @time     : 2018/10/6 19:41
# @file     : logistic_regression.py
# @Software : PyCharm
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))


def logistic_regression(x, y, n):
    w = np.array([1.0, 1.0])
    eta = 1
    b = 0
    itr = 0
    row, column = np.shape(x)
    xpts = np.linspace(-1.5, 2.5)
    print(xpts)
    while itr <= 1000:
        fx = np.dot(w, x.T) + b
        hx = sigmoid(fx)
        t = (hx - y)
        s = [[i[0] * i[1][0], i[0] * i[1][1]] for i in zip(t, x)]
        gradient_w = np.sum(s, 0) / row * eta
        gradient_b = np.sum(t, 0) / row * eta
        w -= gradient_w
        b -= gradient_b
        ypts = (w[0] * xpts + b) / (-w[1])
        itr += 1

    plt.figure()
    for i in range(n):
        plt.plot(x[i, 0], x[i, 1], col[y[i]] + 'o')
    plt.ylim([-1.5, 1.5])
    plt.plot(xpts, ypts, 'r', lw=2)
    plt.title('eta = %s, Iteration = %s\n' % (str(eta), str(itr-1)))
    plt.show()


n = 300
x, y = datasets.make_moons(n, noise=0.3)
col = {0: 'c', 1: 'k'}
logistic_regression(x, y, n)
