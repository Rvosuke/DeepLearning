import numpy as np
from sklearn.datasets import load_iris


def iris_data_load():
    # 加载Iris数据集
    iris = load_iris()
    X, y = iris.data, iris.target

    # 随机打乱数据和标签
    np.random.seed(521)  # 为了可重复性
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    y = y[indices]

    # 划分训练集和测试集（例如，80%训练，20%测试）
    test_size_ratio = 0.2
    test_size = int(X.shape[0] * test_size_ratio)
    X_train = X[:-test_size]
    X_test = X[-test_size:]
    y_train = y[:-test_size]
    y_test = y[-test_size:]

    # 手动实现归一化（特征缩放）
    # 计算训练数据的均值和标准差
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)

    # 对训练数据和测试数据应用归一化
    X_train_scaled = (X_train - mean) / std
    X_test_scaled = (X_test - mean) / std

    return np.float32(X_train_scaled), np.float32(X_test_scaled), y_train, y_test
