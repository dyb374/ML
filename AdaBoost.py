# @Author   : Andrew Dong
# @time     : 2018/12/1 18:48
# @file     : AdaBoost.py
# @Software : PyCharm

import numpy as np
import pandas as pd


def classify(dataMatrix, dimen, thresholdValue, thresholdIneq):
    returnArray = np.ones((np.shape(dataMatrix)[0], 1))
    if thresholdIneq == 'lt':
        returnArray[dataMatrix[:, dimen] <= thresholdValue] = -1
    else:
        returnArray[dataMatrix[:, dimen] > thresholdValue] = -1
    return returnArray


def buildTree(data, classLabels, D):

    dataMatrix = np.mat(data);
    labelMat = np.mat(classLabels).T
    m, n = np.shape(dataMatrix)
    stepNum = 10.0;
    best = {};
    bestClassEst = np.mat(np.zeros((m, 1)))
    minError = np.inf
    for i in range(n):
        rangeMin = dataMatrix[:, i].min();
        rangeMax = dataMatrix[:, i].max()
        stepSize = (rangeMax - rangeMin) / stepNum
        for j in range(-1, int(stepNum) + 1):
            for thresholdIneq in ['lt', 'gt']:
                thresholdValue = rangeMin + float(j) * stepSize
                predictClass = classify(dataMatrix, i, thresholdValue, thresholdIneq)
                errArray = np.mat(np.ones((m, 1)))
                errArray[predictClass == labelMat] = 0
                weightError = D.T * errArray
                if weightError < minError:
                    minError = weightError
                    bestClassEst = predictClass.copy()
                    best['dimen'] = i
                    best['thresholdValue'] = thresholdValue
                    best['thresholdIneq'] = thresholdIneq
    return bestClassEst, minError, best


def adaBoostTrain(data, classLabels, T):
    weakClassArr = []
    m = np.shape(data)[0]
    D = np.mat(np.ones((m, 1)) / m)
    aggClassEst = np.mat(np.zeros((m, 1)))
    # 迭代
    for i in range(T):
        # 调用决策树
        error, classEst, best = buildTree(data, classLabels, D)
        if error > 0.5:
            break
        alpha = float(0.5 * np.log((1.0 - error) / error))
        best['alpha'] = alpha
        weakClassArr.append(best)
        print("classEst:", classEst.T)
        # 为下一次迭代计算D
        exp = np.multiply(-1 * alpha * np.mat(classLabels).T, classEst)
        D = D * exp / D.sum()  # 确保D是一个分布
    return weakClassArr


def adaClassify(data, classifierArr):
    dataMatrix = np.mat(data)
    m = np.shape(dataMatrix)[0]
    aggClassEst = np.mat(np.zeros((m, 1)))
    # 遍历classifierArr中的所有弱分类器，并基于classify对每个分类器得到一个类别的估计值
    for i in range(len(classifierArr)):
        classEst = classify(dataMatrix, classifierArr[i]['dim'], classifierArr[i]['thresh'], classifierArr[i]['ineq'])
        aggClassEst += classifierArr[i]['alpha'] * classEst
    return np.sign(aggClassEst)


d = pd.read_csv("watermelon3.csv", encoding="gb18030")
data = d.values[:, 1:2]
classLabels = d.values[:, 2]

weakClass = adaBoostTrain(data, classLabels, 10)
result = adaClassify(data, weakClass)
