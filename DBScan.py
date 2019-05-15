# @Author   : Andrew Dong
# @time     : 2018/12/10 22:54
# @file     : DBScan.py
# @Software : PyCharm
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.datasets.samples_generator import make_blobs, make_circles


"""
寻找该点的邻居
"""
def find_neighbors(data, pointId, eps):
    nPoints = data.shape[1]
    neighbors = []
    for i in range(nPoints):
        if math.sqrt(np.power(data[:, pointId] - data[:, i], 2).sum()) < eps:
            neighbors.append(i)
    return neighbors


"""
判断是否能成功分类
"""
def expand_cluster(data, clusterResult, pointId, clusterId, eps, minPts):
    seeds = find_neighbors(data, pointId, eps)
    if len(seeds) < minPts:
        clusterResult[pointId] = 0
        return False
    else:
        clusterResult[pointId] = clusterId
        for seedId in seeds:
            clusterResult[seedId] = clusterId

        while len(seeds) > 0:
            currentPoint = seeds[0]
            queryResults = find_neighbors(data, currentPoint, eps)
            if len(queryResults) >= minPts:
                for i in range(len(queryResults)):
                    resultPoint = queryResults[i]
                    if clusterResult[resultPoint] == False:
                        seeds.append(resultPoint)
                        clusterResult[resultPoint] = clusterId
                    elif clusterResult[resultPoint] == 0:
                        clusterResult[resultPoint] = clusterId
            seeds = seeds[1:]
        return True


def dbscan(data, eps, minPts):
    clusterId = 1
    nPoints = data.shape[1]
    clusterResult = [False] * nPoints
    for pointId in range(nPoints):
        point = data[:, pointId]
        if clusterResult[pointId] == False:
            if expand_cluster(data, clusterResult, pointId, clusterId, eps, minPts):
                clusterId = clusterId + 1
    return clusterResult, clusterId - 1


def plotFeature(data, clusters, clusterNum):
    matClusters = np.mat(clusters).transpose()
    fig = plt.figure()
    scatterColors = ['black', 'blue', 'green', 'yellow', 'red', 'purple', 'orange', 'brown']
    ax = fig.add_subplot(111)
    for i in range(clusterNum + 1):
        colorSytle = scatterColors[i % len(scatterColors)]
        subCluster = data[:, np.nonzero(matClusters[:, 0].A == i)]
        ax.scatter(subCluster[0, :].flatten().A[0], subCluster[1, :].flatten().A[0], c=colorSytle, s=50)


X1, y1 = make_blobs(n_samples=100, n_features=2, centers=[[1.5, 1.5]], cluster_std=[[0.1]], random_state=5)
X2, y2 = make_circles(n_samples=200, factor=0.6, noise=0.05, random_state=1)
X = np.concatenate((X1, X2))
X = np.mat(X).transpose()

clusters, clusterNum = dbscan(X, 0.2, 10)
print("聚类数量:", clusterNum)

plotFeature(X, clusters, clusterNum)
plt.show()
