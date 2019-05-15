# @Author   : Andrew Dong
# @time     : 2018/11/26 22:31
# @file     : Bayesian_network_EM.py
# @Software : PyCharm
import pandas as pd
import numpy as np

"""
X为可被观察数据
A为模型参数矩阵
tol为阈值
iterations为迭代最大次数
"""
def em(X, A, tol, iterations):

    i = 0
    while i < iterations:
        new_A = em_single(X, A)
        change = np.abs(A - new_A)
        if change < tol:
            break
        else:
            A = new_A
            i += 1

    return new_A


def em_single(X, A):

    #E step
    #Z对应脐部的3中状态:凹陷，稍凹，平坦
    Z = exceptation(X, A)
    #M step
    new_A = maximization(X, Z)
    return new_A

def exceptation(X, A):
    Z = 0
    #求出隐变量的分布Z
    return Z

def maximization(X, Z):
    new_A = 0
    #最大化期望似然
    return new_A


X = pd.read_csv("watermelon.csv", encoding="gb18030")
A = np.eye(6)#最多一个属性可跟6个属性关联
print("模型参数为:" + em(X, A, 10000))


