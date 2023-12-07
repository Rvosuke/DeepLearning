import numpy as np
from typing import List, Tuple, Union


class KernelSVM:
    """
    Kernel Support Vector Machine (Kernel SVM)类
    """

    def __init__(self, kernel: str = 'rbf', C: float = 1.0, max_iter: int = 1000, tol: float = 1e-3) -> None:
        """
        初始化函数

        Args:
            kernel (str): 核函数类型，'linear'或'rbf'。
            C (float): 惩罚参数。
            max_iter (int): 最大迭代次数。
            tol (float): 精度控制参数。
        """
        self.b = None
        self.sv_idx = None
        self.support_vectors_ = None
        self.support_vector_labels_ = None
        self.alpha = None
        self.kernel = kernel
        self.C = C
        self.max_iter = max_iter
        self.tol = tol

    @staticmethod
    def _linear_kernel(x1: np.ndarray, x2: np.ndarray) -> float:
        """
        线性核函数

        Args:
            x1, x2 (np.ndarray): 两个样本点。

        Returns:
            核函数计算结果 (float)
        """
        return np.dot(x1, x2)

    @staticmethod
    def _rbf_kernel(x1: np.ndarray, x2: np.ndarray, sigma: float = 5.0) -> float:
        """
        高斯径向基（RBF）核函数

        Args:
            x1, x2 (np.ndarray): 两个样本点。
            sigma (float): 高斯函数的标准差。

        Returns:
            核函数计算结果 (float)
        """
        return np.exp(-np.linalg.norm(x1 - x2) ** 2 / (2 * (sigma ** 2)))

    def _compute_kernel(self, X: np.ndarray) -> np.ndarray:
        """
        计算核矩阵

        Args:
            X (np.ndarray): 输入样本。

        Returns:
            核矩阵 (np.ndarray)
        """
        n_samples, n_features = X.shape
        K = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                if self.kernel == 'linear':
                    K[i, j] = self._linear_kernel(X[i], X[j])
                elif self.kernel == 'rbf':
                    K[i, j] = self._rbf_kernel(X[i], X[j])
        return K

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        使用输入的训练数据训练模型。训练过程使用SMO(Sequential Minimal Optimization)算法进行参数优化。

        Args:
            X (np.ndarray): 输入样本。形状为 [n_samples, n_features] 的二维数组，n_samples 为样本数量，n_features 为特征数量。
            y (np.ndarray): 样本标签。形状为 [n_samples] 的一维数组。

        训练步骤：
            1. 初始化：获取样本和特征数量，计算核矩阵，初始化参数 alpha 和偏置项 b。
            2. 主训练循环：对每一个样本，根据SMO算法更新 alpha 和 b。
            3. 检查收敛：如果新旧 alpha 之间的欧几里得范数小于给定的阈值(tol)，则认为算法已经收敛，停止循环。
            4. 保存支持向量：所有 alpha 大于 0 的样本被视为支持向量，这些支持向量和对应的 alpha 将用于后续的预测任务。

        注意：
            虽然可以直接调用 fit 函数来训练模型，但是为了获取最好的性能，通常还需要对数据进行适当的预处理，
            并且可能需要调整模型的参数（C，kernel，max_iter 和 tol等）。

        """

        # 获取样本数量和特征数量
        n_samples, n_features = X.shape

        # 计算核矩阵
        K = self._compute_kernel(X)

        # 初始化参数alpha和偏置项b
        self.alpha = np.zeros(n_samples)
        self.b = 0

        # 主训练循环
        for _ in range(self.max_iter):
            # 备份当前的alpha
            alpha_prev = np.copy(self.alpha)

            for j in range(n_samples):
                # 随机选择一个不同于j的索引i
                i = self._get_random_int(j, n_samples)

                # 计算eta，用于后续的alpha更新
                eta = 2.0 * K[i, j] - K[i, i] - K[j, j]

                # 如果eta大于或等于0，跳过当前的j
                if eta >= 0:
                    continue

                # 更新alpha[j]
                self.alpha[j] -= (
                        y[j] * (K[i, j] - self.b - y[i] * (K[i, j] - self.b) - y[j] * (K[j, j] - self.b)) / eta)

                # 确保alpha[j]在[0, C]范围内
                self.alpha[j] = max(0, min(self.alpha[j], self.C))

                # 更新偏置项b
                self.b = self.b - y[i] * (self.alpha[i] - alpha_prev[i]) * K[i, i] - y[j] * (
                        self.alpha[j] - alpha_prev[j]) * K[i, j]

            # 检查是否收敛
            if np.linalg.norm(self.alpha - alpha_prev) < self.tol:
                break

        # 保存所有alpha大于0的样本，它们是支持向量
        self.sv_idx = np.where(self.alpha > 0)[0]
        self.support_vectors_ = X[self.sv_idx]
        self.support_vector_labels_ = y[self.sv_idx]
        self.alpha = self.alpha[self.sv_idx]

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        使用训练好的模型进行预测。

        Args:
            X (np.ndarray): 输入样本。

        Returns:
            预测结果 (np.ndarray)
        """
        y_predict = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            s = 0
            for a, sv_y, sv in zip(self.alpha, self.support_vector_labels_, self.support_vectors_):
                if self.kernel == 'linear':
                    s += a * sv_y * self._linear_kernel(X[i], sv)
                elif self.kernel == 'rbf':
                    s += a * sv_y * self._rbf_kernel(X[i], sv)
            y_predict[i] = s
        return np.sign(np.intp(y_predict + self.b))

    @staticmethod
    def _get_random_int(i: int, max_value: int) -> int:
        """
        生成一个不等于i的随机整数。

        Args:
            i (int): 需要避免的整数。
            max_value (int): 随机整数的上限。

        Returns:
            随机生成的整数 (int)
        """
        j = i
        while j == i:
            j = np.random.randint(max_value)
        return j


if __name__ == '__main__':
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_curve, auc, accuracy_score
    from Algrithm.relief import Relief
    from matplotlib import pyplot as plt

    # 加载数据集
    data = load_breast_cancer()

    # 数据预处理
    X, y = data.data, data.target
    y = np.where(y == 0, -1, y)  # 将类别标签从 {0,1} 转化为 {-1,1}，以适应我们的SVM模型
    X = StandardScaler().fit_transform(X)  # 标准化数据

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    relief = Relief(n_features=10)
    X_train, X_test = relief.fit_transform(X_train, y_train), relief.transform(X_test)

    # 创建模型并进行训练
    model = KernelSVM(kernel='linear', C=1.0, max_iter=1000, tol=1e-3)
    model.fit(X_train, y_train)

    # 在测试集上进行预测
    y_pred = model.predict(X_test)

    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    roc_auc = auc(fpr, tpr)

    # 打印预测结果
    print(np.intp(accuracy_score(y_test, y_pred)*100))
    print("Unique values in predictions: ", np.unique(y_pred))
    print("Unique values in test labels: ", np.unique(y_test))
    plt.figure()
    plt.plot(fpr, tpr, label='Mean ROC (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Mean ROC')
    plt.legend(loc="lower right")
    plt.show()
