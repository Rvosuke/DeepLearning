import numpy as np
from collections import defaultdict
from typing import List, Tuple


class AODE:

    def __init__(self, min_samples: int = 1):
        self.class_freqs = defaultdict(int)  # 用于存储类别的频率
        self.cond_freqs = defaultdict(int)  # 用于存储条件频率
        self.min_samples = min_samples  # 设定的最小样本数，对于一个属性，如果它在某个类别下的样本数少于该值，则不使用它来做预测

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        使用输入的训练数据训练模型。

        Args:
            X (np.ndarray): 输入样本。
            y (np.ndarray): 样本标签。
        """
        # 遍历数据，计算类别的频率和条件频率
        for xi, yi in zip(X, y):
            self.class_freqs[yi] += 1
            for attr, val in enumerate(xi):
                self.cond_freqs[(attr, val, yi)] += 1

    def predict(self, X: np.ndarray) -> List:
        """
        使用模型对输入样本进行预测。

        Args:
            X (np.ndarray): 输入样本。

        Returns:
            List: 预测结果列表。
        """
        # 对每个样本，计算每个类别的概率，并选择最大的
        y_pred = []
        for xi in X:
            max_prob = -1
            max_class = None
            for yi in self.class_freqs.keys():
                if self.class_freqs[yi] >= self.min_samples:
                    prob = self.class_freqs[yi]
                    for attr, val in enumerate(xi):
                        prob *= self.cond_freqs.get((attr, val, yi), 1)
                    if prob > max_prob:
                        max_prob = prob
                        max_class = yi
            y_pred.append(max_class)
        return y_pred


if __name__ == '__main__':
    # 使用示例：
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split

    # 加载数据
    data = load_breast_cancer()
    X = data.data
    y = data.target

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 训练模型
    aode = AODE()
    aode.fit(X_train, y_train)

    # 预测
    y_pred = aode.predict(X_test)

    # 输出预测准确率
    print("Accuracy: ", np.mean(y_pred == y_test))
