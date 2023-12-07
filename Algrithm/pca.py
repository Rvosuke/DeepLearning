import numpy as np
from typing import Tuple


class PCA:
    def __init__(self, n_components: int):
        self.n_components = n_components  # 主成分数
        self.components = None  # 主成分

    def fit(self, X: np.ndarray) -> None:
        """
        使用输入的训练数据训练PCA模型。

        Args:
            X (np.ndarray): 输入样本。
        """
        # 数据中心化
        X = X - np.mean(X, axis=0)

        # 计算协方差矩阵
        cov = np.cov(X.T)

        # 计算特征值和特征向量
        eigenvalues, eigenvectors = np.linalg.eig(cov)

        # 降序排列特征值，并选择前n_components个特征向量
        idx = eigenvalues.argsort()[::-1]
        eigenvectors = eigenvectors[:, idx]
        self.components = eigenvectors[:, :self.n_components]

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        使用PCA模型转换输入样本。

        Args:
            X (np.ndarray): 输入样本。

        Returns:
            np.ndarray: 转换后的样本。
        """
        # 数据中心化
        X = X - np.mean(X, axis=0)

        # 使用主成分进行线性变换
        return np.dot(X, self.components)
