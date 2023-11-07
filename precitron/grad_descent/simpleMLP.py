import numpy as np
from typing import Tuple


def sigmoid(x: np.ndarray) -> np.ndarray:
    """
    sigmoid函数
    :param x:
    :return: 1 / (1 + e^(-x))
    """
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x: np.ndarray) -> np.ndarray:
    """
    sigmoid函数的导数
    :param x:
    :return: y(1-y)
    """
    return sigmoid(x) * (1 - sigmoid(x))


def tanh(x: np.ndarray) -> np.ndarray:
    """
    双曲正切函数
    :param x:
    :return: e^x - e^(-x) / e^x + e^(-x)
    """
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))


def tanh_derivative(x: np.ndarray) -> np.ndarray:
    """
    双曲正切函数的导数
    :param x:
    :return: 1 - y^2
    """
    return 1 - tanh(x) ** 2


def binary_cross_entropy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    二元交叉熵损失函数
    :param y_true: 真实值
    :param y_pred: 预测值
    :return: -ylog(y') - (1-y)log(1-y')
    """
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)  # 防止log(0)的情况, clip函数将数组中的元素限制在a_min, a_max之间
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


def binary_cross_entropy_derivative(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    二元交叉熵损失函数的导数
    :param y_true: 真实值
    :param y_pred: 预测值
    :return: -y/y'+(1-y)/(1-y')
    """
    return -(y_true / y_pred) + (1 - y_true) / (1 - y_pred)


class MLP:
    def __init__(self, input_size: int, hidden_sizes: Tuple[int, int], output_size: int):
        # 初始化权重和偏置, 从标准正态分布中返回一个或多个样本值
        self.weights_input_to_hidden1 = np.random.randn(input_size, hidden_sizes[0])
        self.bias_hidden1 = np.random.randn(hidden_sizes[0])

        self.weights_hidden1_to_hidden2 = np.random.randn(hidden_sizes[0], hidden_sizes[1])
        self.bias_hidden2 = np.random.randn(hidden_sizes[1])

        self.weights_hidden2_to_output = np.random.randn(hidden_sizes[1], output_size)
        self.bias_output = np.random.randn(output_size)

        # 重命名, 便于阅读
        self.w1 = self.weights_input_to_hidden1
        self.b1 = self.bias_hidden1
        self.w2 = self.weights_hidden1_to_hidden2
        self.b2 = self.bias_hidden2
        self.w3 = self.weights_hidden2_to_output
        self.b3 = self.bias_output

    def forward(self, x: np.ndarray) -> np.ndarray:
        # 正向传播
        self.h1 = tanh(np.dot(x, self.w1) + self.b1)  # hidden1 = tanh(xw1+b1)
        self.h2 = tanh(np.dot(self.h1, self.w2) + self.b2)  # hidden2 = tanh(h1w2+b2)
        p = sigmoid(np.dot(self.h2, self.w3) + self.b3)  # precision = sigmoid(h2w3+b3)
        return p

    def backward(self, x: np.ndarray, y: np.ndarray, output: np.ndarray):
        # 反向传播, 采用随机梯度下降算法
        e = binary_cross_entropy(y, output)  # e = -ylog(p)-(1-y)log(1-p)
        w3_delta = binary_cross_entropy_derivative(y, output) * sigmoid_derivative(output) * self.w3  # w3'=e'p'y'
        w2_delta = w3_delta.dot(self.w3.T) * tanh_derivative(self.h2) * self.w2  # w2'=w3'w3tanh'(h2)w2
        w1_delta = w2_delta.dot(self.w2.T) * tanh_derivative(self.h1) * self.w1  # w1'=w2'w2tanh'(h1)w1
        b3_delta = binary_cross_entropy_derivative(y, output) * sigmoid_derivative(output) * self.b3  # b3'=e'p'y'
        b2_delta = w3_delta.dot(self.w3.T) * tanh_derivative(self.h2) * self.b2  # b2'=w3'w3tanh'(h2)b2
        b1_delta = w2_delta.dot(self.w2.T) * tanh_derivative(self.h1) * self.b1  # b1'=w2'w2tanh'(h1)b1
        parameters = [w3_delta, w2_delta, w1_delta, b3_delta, b2_delta, b1_delta]
        return parameters

    def optimize(self, parameters: list, learning_rate: float):
        # 更新权重和偏置
        self.w3 -= learning_rate * parameters[0]
        self.w2 -= learning_rate * parameters[1]
        self.w1 -= learning_rate * parameters[2]
        self.b3 -= learning_rate * parameters[3]
        self.b2 -= learning_rate * parameters[4]
        self.b1 -= learning_rate * parameters[5]

    def train(self, x: np.ndarray, y: np.ndarray, epochs: int, learning_rate: float):
        for epoch in range(1, epochs + 1):
            # 正向传播
            output = self.forward(x)
            # 反向传播
            parameters = self.backward(x, y, output)
            # 更新权重和偏置
            self.optimize(parameters, learning_rate)
            print(f'Epoch: {epoch}, Loss: {binary_cross_entropy(y, output):.2f}')

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)

# 示例用法:
# mlp = MLP(input_size=4, hidden_sizes=(10, 5), output_size=1)
# mlp.train(X_train, y_train, epochs=1000, learning_rate=0.1)
# predictions = mlp.predict(X_test)
import sklearn

# 导入iris
iris = sklearn.datasets.load_iris()
X = iris.data
y = iris.target
# 将y转换为one-hot编码
y_one_hot = np.zeros((y.shape[0], 3))
y_one_hot[np.arange(y.shape[0]), y] = 1
# 划分训练集和测试集
from sklearn.model_selection import train_test_split
train_set = train_test_split(X, y_one_hot, test_size=0.2, random_state=42)
X_train, X_test, y_train, y_test = train_set
# 创建MLP
mlp = MLP(input_size=4, hidden_sizes=(10, 5), output_size=3)
# 训练
mlp.train(X_train, y_train, epochs=1000, learning_rate=0.1)
# 预测
predictions = mlp.predict(X_test)
# 评估
from sklearn.metrics import accuracy_score
print(f'Accuracy: {accuracy_score(np.argmax(y_test, axis=1), np.argmax(predictions, axis=1)):.2f}')

