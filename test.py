# @Author   : Andrew Dong
# @time     : 2018/12/11 8:36
# @file     : test.py
# @Software : PyCharm
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.datasets.samples_generator import make_circles



def eps_neighbor(a, b, eps):
    return math.sqrt(np.power(a - b, 2).sum()) < eps

def find_neighbor(data, pointId, eps):
    nPoints = data.shape[1]
    neighbors = []
    for i in range(nPoints):
        if eps_neighbor(data[:, pointId], data[:, i], eps):
            neighbors.append(i)
    return neighbors

def expand_cluster(data, clusterResult, pointId, clusterId, eps, minPts):
    neighbors = find_neighbor(data, pointId, eps)
    if len(neighbors) < minPts:
        clusterResult[pointId] = 0
        return False
    else:
        clusterResult[pointId] = clusterId
        for seedId in neighbors:
            clusterResult[seedId] = clusterId

        while len(neighbors) > 0:
            currentPoint = neighbors[0]
            queryResults = find_neighbor(data, currentPoint, eps)
            if len(queryResults) >= minPts:
                for i in range(len(queryResults)):
                    resultPoint = queryResults[i]
                    if clusterResult[resultPoint] == False:
                        neighbors.append(resultPoint)
                        clusterResult[resultPoint] = clusterId
                    elif clusterResult[resultPoint] == 0:
                        clusterResult[resultPoint] = clusterId
            neighbors = neighbors[1:]
        return True

def dbscan(data, eps, minPts):
    clusterId = 1
    nPoints = data.shape[1]
    clusterResult = [False] * nPoints
    for pointId in range(nPoints):
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

dataSet, y = make_circles(n_samples=200, factor=0.6, noise=0.05, random_state=1)
dataSet = np.mat(dataSet).transpose()
clusters, clusterNum = dbscan(dataSet, 0.2, 10)
plotFeature(dataSet, clusters, clusterNum)

plt.show()

