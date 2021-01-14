import pandas as pd
import numpy as np


def data_analysis(data):
    """
    连续数据的分析
    :param data: 数据集
    :return:
    """
    means = np.mean(data, axis=0)
    medians = np.median(data, axis=0)
    sigma = np.std(data, axis=0, ddof=1)
    a = np.max(data, axis=0) - np.min(data, axis=0)
    int_r = np.quantile(data, 0.75, axis=0, interpolation='higher') - np.quantile(data, 0.25, axis=0,
                                                                                  interpolation='lower')
    print("均值：", means)
    print("中位数：", medians)
    print("标准差：", sigma)
    print("极差：", a)
    print("四分位距：", int_r)


def to_categorical(x, n_col=None):
    if not n_col:
        n_col = np.amax(x) + 1
    one_hot = np.zeros((x.shape[0], n_col))
    one_hot[np.arange(x.shape[0]), x] = 1
    return one_hot


class Normalization:
    """
    归一化
    """

    def __init__(self):
        self.range = None
        self.min = None

    def fit_transform(self, data):
        self.range = np.max(data) - np.min(data)
        self.min = np.min(data)
        return (data - self.min) / self.range

    def inverse_transform(self, data):
        assert self.range is not None and self.min is not None
        return self.min + data * self.range


class Standardization:
    """
    数据标准化
    """

    def __init__(self):
        self.mean = None
        self.sigma = None

    def fit_transform(self, data):
        self.mean = np.mean(data, axis=0)
        self.sigma = np.std(data, axis=0, ddof=1)
        return (data - self.mean) / self.sigma

    def inverse_transform(self, data):
        assert self.mean is not None and self.sigma is not None
        return self.mean + (data * self.sigma)


def fill(data):
    """
    简单的缺失值处理
    """
    for i in range(data.shape[1]):
        temp_col = data[:, i]
        nan_num = np.count_nonzero(temp_col != temp_col)
        if nan_num != 0:
            temp_not_nan_col = temp_col[temp_col == temp_col]
            temp_col[np.isnan(temp_col)] = temp_not_nan_col.mean()
    return data


class find_error_data:
    def __init__(self, data):
        """
        异常值检测
        :param data:
        """
        self.error_data = []  # 记录异常值的位置
        self.data = data

    def find_error_var(self):
        for i in range(self.data.shape[1]):
            temp_col = self.data[:, i]
            sigma = np.std(temp_col, ddof=1)
            mean = np.mean(temp_col)
            for j in range(len(temp_col)):
                if mean - 3 * sigma > temp_col[j] or mean + 3 * sigma < temp_col[j]:
                    self.error_data.append([j, i])

    def find_error_box(self):
        for i in range(self.data.shape[1]):
            temp_col = self.data[:, i]
            Q = np.percentile(temp_col, [25, 75])
            IQR = Q[1] - Q[0]
            up = Q[1] + 1.5 * IQR
            down = Q[0] - 1.5 * IQR
            for j in range(len(temp_col)):
                if temp_col[j] > up or temp_col[j] < down:
                    self.error_data.append([j, i])

    def deal_error(self):
        for i in self.error_data:
            x = i[0]
            y = i[1]
            self.data[x][y] = (np.sum(self.data[:, y]) - self.data[x][y]) / (self.data.shape[0] - 1)  # 对异常值填充均值处理


def resemble_data(data1, data2):
    """
    数据相似性分析
    :param data1: 数据1
    :param data2: 数据2
    :return:
    """
    assert data1.shape == data2.shape
    Euclidean = np.linalg.norm(data1 - data2)
    Manhattan = np.linalg.norm(data1 - data2, ord=1)
    cos = data1.dot(data2) / (np.linalg.norm(data1) * np.linalg.norm(data2))
    Pearson = np.corrcoef(data1, data2)
    print("欧氏距离：", Euclidean)
    print("曼哈顿距离：", Manhattan)
    print("余弦值：", cos)
    print("皮尔逊相关系数：", Pearson)
    return Euclidean, Manhattan, cos, Pearson


def group(data, feature, way='mean'):
    """
    聚合与分组
    :param data: 数据集
    :param feature: 对哪一个特征进行分组
    :param way: 聚合标准
    :return:
    """
    data = pd.DataFrame(data)
    if way == 'mean':
        data = data.groupby(feature).mean()
    elif way == 'sum':
        data = data.groupby(feature).sum()
    elif way == 'median':
        data = data.groupby(feature).std()
    return data


def moving_average(data, n):  # 没有理解透彻
    """
    数据的平滑处理，滑动平均法
    :param data: 数据集
    :param n: 取平均值队列大小
    :return:
    """
    window = np.ones(int(n)) / float(n)
    return np.convolve(data, window, 'same')  # numpy的卷积函数


def VarianceThreshold(data, threshold=1):
    """
    特征值选择 方差选择法
    :param data: 数据集
    :param threshold: 方差阈值
    :return: 筛选后的数据集
    """
    var = np.var(data, axis=0, ddof=1)
    count = []
    for i in range(len(var)):
        if var[i] < threshold: count.append(i)
    data = np.delete(data, count, axis=1)
    return data


def chi(X, y):
    """
    卡方检验
    :param X: 特征集
    :param y: 结果集
    :return:
    """
    res = np.empty((X.shape[1],))
    for i in range(X.shape[1]):
        a = X[:, i]
        N01 = np.sum(a == 0 and y == 1)
        N11 = np.sum(a == 1 and y == 1)
        N10 = np.sum(a == 1 and y == 0)
        N00 = np.sum(a == 0 and y == 0)
        res[i] = (X.shape[0] * (N11 * N00 - N10 * N01) ** 2) / ((N01 + N00) * (N11 + N10) * (N11 + N01) * (N10 + N00))
    return res
#
# if __name__ == '__main__':
#     cxk = Standardization()
#     X = np.arange(24).reshape(6,4)
#     b = cxk.fit_transform(X)
#     print(b)
#     print("---"*20)
#     print(cxk.inverse_transform(b))
