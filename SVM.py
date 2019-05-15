# @Author   : Andrew Dong
# @time     : 2018/12/24 21:14
# @file     : SVM.py
# @Software : PyCharm
import random
import math

EPS = 0.000000001


def load_data(filename, data, dim):
    for line in open(filename, 'rt'):
        sample = [0.0 for v in range(0, dim + 1)]
        line = line.rstrip("\r\n\t ")
        fields = line.split("\t")
        label = int(fields[0])
        sample[0] = 1.0
        for field in fields[1:]:
            kv = field.split(":")
            idx = int(kv[0])
            val = float(kv[1])
            sample[idx] = val
        data.append((label, sample))


def svm_train(train_data, dim, W, iterations, lm, lr):
    X = [0.0 for v in range(0, dim + 1)];  # <sample, label> => <X, y>
    grad = [0.0 for v in range(0, dim + 1)];  # 梯度
    num_train = len(train_data);
    for i in range(0, iterations):
        # 每次迭代随机选择一个训练样本
        index = random.randint(0, num_train - 1)
        y = train_data[index][0]
        for j in range(0, dim + 1):
            X[j] = train_data[index][1][j]
        # 计算梯度
        WX = 0.0
        for j in range(0, dim + 1):
            WX += W[j] * X[j]
        if 1 - WX * y > 0:
            for j in range(0, dim + 1):
                grad[j] = lm * W[j] - X[j] * y
        else:  # 1-WX *y <= 0的时候，目标函数的前半部分恒等于0, 梯度也是0
            for j in range(0, dim + 1):
                grad[j] = lm * W[j] - 0

        # 更新权重, lr是学习速率
        for j in range(0, dim + 1):
            W[j] = W[j] - lr * grad[j]


def svm_predict(test_data, dim, W):
    num_test = len(test_data)
    num_correct = 0
    for i in range(0, num_test):
        target = test_data[i][0]  # 即label
        X = test_data[i][1]  # 即sample
        sum = 0.0
        for j in range(0, dim + 1):
            sum += X[j] * W[j]
        predict = -1
        if sum > 0:  # 权值>0，认为目标值为1
            predict = 1
        if predict * target > 0:  # 预测值和目标值符号相同
            num_correct += 1

    return num_correct * 1.0 / num_test;


if __name__ == "__main__":
    # 设置参数
    epochs = 100  # 迭代轮数
    iterations = 10  # 每一轮中梯度下降迭代次数, 这个其实可以和epochs合并为一个参数
    train_data = []  # 训练集, 假设每个样本的特征数量都一样
    test_data = []  # 测试集, 假设每个样本的特征数量都一样
    lm = 0.0001  # lambda, 对权值做正则化限制的权重
    lr = 0.01  # lr, 是学习速率，用于调整训练收敛的速度
    dim = 1000  # dim, 特征的最大维度, 所有样本不同特征总数
    W = [0.0 for v in range(0, dim + 1)]
    # 导入测试集&训练集
    load_data("train.txt", train_data, dim)
    load_data("test.txt", test_data, dim)
    # 训练, 实际迭代次数=epochs * iterations
    for i in range(0, epochs):
        svm_train(train_data, dim, W, iterations, lm, lr)
        accuracy = svm_predict(test_data, dim, W)
        print("epoch:%d\t accuracy:%f" % (i, accuracy))
    # 输出结果权值
    for i in range(0, dim + 1):
        if math.fabs(W[i]) > EPS:
            print("%d\t%f" % (i, W[i]))
