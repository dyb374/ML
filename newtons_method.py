# @Author   : Andrew Dong
# @time     : 2018/10/6 9:52
# @file     : newtons_method.py
# @Software : PyCharm
from sympy import *


def newtons_method(f, e):
    x = Symbol("x")

    '''求一阶导数'''
    f1 = diff(f, x)

    def h(x):
        return eval(str(f))

    def h1(x):
        return eval(str(f1))

    '''令迭代初始值xk为0'''
    xk = 0
    k = 1
    while abs(h(xk)) > e:
        xk -= h(xk)/h1(xk)
        k += 1

    print("迭代次数为:", k, "次")
    print("当x为", xk, "时, 极小值为:", h(xk))


newtons_method("x*x*x + 2*x*x +3*x + 4", 0.0001)
