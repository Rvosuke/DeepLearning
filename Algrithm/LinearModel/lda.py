import numpy as np
from typing import List


class LinearDiscriminantAnalysis:
    def __init__(self, dataset: np.ndarray, labels: List[int]):
        """
        线性判别分析构造函数

        Args:
            dataset (np.ndarray): 输入样本。
            labels (List[int]): 样本的类别标签。
        """
        self.dataset = dataset
        self.labels = labels
        self.classes = np.unique(labels)
        self.mean_vectors = None
        self.within_scatter_matrix = None
        self._weights = None

    def _mean_vector(self):
        """
        计算每个类别的均值向量。

        Returns:
            ndarray: 各类别的均值向量数组。
        """
        if self.mean_vectors is None:
            self.mean_vectors = np.array([
                self.dataset[self.labels == cls].mean(axis=0)
                for cls in self.classes
            ])
        return self.mean_vectors

    def _within_scatter(self):
        """
        计算类内散布矩阵。

        Returns:
            ndarray: 类内散布矩阵。
        """
        if self.within_scatter_matrix is None:
            cov_matrices = [
                np.cov((self.dataset[self.labels == cls] - mean_vec).T, bias=False)
                for cls, mean_vec in zip(self.classes, self._mean_vector())
            ]
            self.within_scatter_matrix = np.sum(cov_matrices, axis=0)
        return self.within_scatter_matrix

    def _weights(self):
        """
        计算权重。

        Returns:
            ndarray: 权重数组。
        """
        if self._weights is None:
            mean_diff = self._mean_vector()[1] - self._mean_vector()[0]
            self._weights = np.linalg.inv(self._within_scatter()).dot(mean_diff)
        return self._weights

    def fit(self) -> np.ndarray:
        """
        训练模型。

        Returns:
            ndarray: 权重数组。
        """
        return self._weights()
