# @Author   : Andrew Dong
# @time     : 2018/11/17 20:40
# @file     : LaplacianNB.py
# @Software : PyCharm

import pandas as pd
import math


class LaplacianNBC:

    def train(self, X, y):

        # N为样本数
        N = len(y)

        classes = {}
        for i in y:
            if i in classes:
                classes[i] += 1
            else:
                classes[i] = 1
        # class_n为类别数
        class_n = len(classes)
        # class_p为P(c)
        self.class_p = {}

        for c, n in classes.items():
            self.class_p[c] = float(n + 1) / (N + class_n)

        # 用于存储离散属性
        self.good_discrete_attrs = []
        self.bad_discrete_attrs = []

        # 用于存储连续属性
        self.good_means = []
        self.good_vars = []
        self.bad_means = []
        self.bad_vars = []

        for i in range(6):
            is_good = []
            is_bad = []
            for j in range(N):
                if y[j] == "是":
                    is_good.append(X[j][i])
                else:
                    is_bad.append(X[j][i])

            unique_with_good = {}
            for i in is_good:
                if i in unique_with_good:
                    unique_with_good[i] += 1
                else:
                    unique_with_good[i] = 1

            unique_with_bad = {}
            for i in is_bad:
                if i in unique_with_bad:
                    unique_with_bad[i] += 1
                else:
                    unique_with_bad[i] = 1
            good_d = {}
            for a, n in unique_with_good.items():
                good_d[a] = float(n + 1) / (classes["是"] + len(unique_with_good))
            self.good_discrete_attrs.append(good_d)
            bad_d = {}
            for a, n in unique_with_bad.items():
                bad_d[a] = float(n + 1) / (classes["否"] + len(unique_with_bad))
            self.bad_discrete_attrs.append(bad_d)

        for i in range(2):
            is_good = []
            is_bad = []
            for j in range(N):
                if y[j] == "是":
                    is_good.append(X[j][i + 6])
                else:
                    is_bad.append(X[j][i + 6])
            # 计算均值与方差
            good_mean = sum(is_good) / float(len(is_good))
            good_var = 0
            for i in range(len(is_good)):
                good_var += (is_good[i] - good_mean) ** 2
            good_var = good_var / float(len(is_good))

            bad_mean = sum(is_bad) / float(len(is_bad))
            bad_var = 0
            for i in range(len(is_bad)):
                bad_var += (is_bad[i] - bad_mean) ** 2
            bad_var = bad_var / float(len(is_bad))

            self.good_means.append(good_mean)
            self.good_vars.append(good_var)
            self.bad_means.append(bad_mean)
            self.bad_vars.append(bad_var)

    def predict(self, x):

        good = self.class_p["是"]
        bad = self.class_p["否"]
        for i in range(6):
            good *= self.good_discrete_attrs[i][x[i]]
            bad *= self.bad_discrete_attrs[i][x[i]]
        for i in range(2):
            good *= 1.0 / (math.sqrt(2 * math.pi) * math.sqrt(self.good_vars[i])) * math.exp(- (x[i + 6] - self.good_means[i]) ** 2 / (2 * self.good_vars[i]))
            bad *= 1.0 / (math.sqrt(2 * math.pi) * math.sqrt(self.bad_vars[i])) * math.exp(- (x[i + 6] - self.bad_means[i]) ** 2 / (2 * self.bad_vars[i]))
        if good >= bad:
            return good, bad, "是"
        else:
            return good, bad, "否"


if __name__ == "__main__":
    lnb = LaplacianNBC()
    data = pd.read_csv("watermelon3.csv", encoding="gb18030")
    X = data.values[:, 1:9]
    y = data.values[:, 9]

    lnb.train(X, y)

    label = lnb.predict(["青绿", "蜷缩", "浊响", "清晰", "凹陷", "硬滑", 0.697, 0.460])

    print("预测结果: ", label)
