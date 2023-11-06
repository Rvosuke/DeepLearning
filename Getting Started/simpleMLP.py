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
        # 反向传播
        e = binary_cross_entropy(y, output)  # e = -ylog(p)-(1-y)log(1-p)
        w3_delta = binary_cross_entropy_derivative(y, output) * sigmoid_derivative(output) * self.w3  # w3'=e'p'y'
        output_error = binary_cross_entropy(y, output)
        output_delta = output_error * sigmoid_derivative(output)  # p'=y(1-y)y'=sigmoid_derivative(output)*w

        hidden2_error = output_delta.dot(self.weights_hidden2_to_output.T)
        hidden2_delta = hidden2_error * sigmoid_derivative(self.hidden2)

        hidden1_error = hidden2_delta.dot(self.weights_hidden1_to_hidden2.T)
        hidden1_delta = hidden1_error * sigmoid_derivative(self.hidden1)

        return ...

        # 更新权重和偏置
        self.weights_hidden2_to_output += self.hidden2.T.dot(output_delta)
        self.bias_output += np.sum(output_delta, axis=0)

        self.weights_hidden1_to_hidden2 += self.hidden1.T.dot(hidden2_delta)
        self.bias_hidden2 += np.sum(hidden2_delta, axis=0)

        self.weights_input_to_hidden1 += x.T.dot(hidden1_delta)
        self.bias_hidden1 += np.sum(hidden1_delta, axis=0)

    def train(self, x: np.ndarray, y: np.ndarray, epochs: int, learning_rate: float):
        for epoch in range(epochs):
            output = self.forward(x)
            self.backward(x, y, output)
            loss = binary_cross_entropy(y, output)
            print(f'Epoch {epoch + 1}/{epochs} loss: {loss:.4f}')

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)

# 示例用法:
# mlp = MLP(input_size=4, hidden_sizes=(10, 5), output_size=1)
# mlp.train(X_train, y_train, epochs=1000, learning_rate=0.1)
# predictions = mlp.predict(X_test)
