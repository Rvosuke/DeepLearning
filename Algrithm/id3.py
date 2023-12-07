from typing import Union

import numpy as np
import pandas as pd


class ID3:
    """
    ID3算法实现的决策树类
    """

    def __init__(self, target: str) -> None:
        """
        初始化函数

        Args:
            target (str): 目标变量的名称
        """
        self.target = target

    def fit(self, data: pd.DataFrame) -> dict:
        """
        使用输入的训练数据训练模型。

        Args:
            data (pd.DataFrame): 输入样本。

        Returns:
            决策树 (dict)
        """
        features = [col for col in data.columns if col != self.target]
        return self._id3(data, features, None)

    def _id3(self, data: pd.DataFrame, features: list, parent_data: pd.DataFrame) -> Union[str, dict]:
        """
        ID3算法实现

        Args:
            data (pd.DataFrame): 输入样本。
            features (list): 特征列表。
            parent_data (pd.DataFrame): 父节点数据。

        Returns:
            决策树或者类别 (Union[str, dict])
        """
        target_values = data[self.target].unique()

        if len(target_values) == 1:
            return target_values[0]
        elif len(features) == 0 or len(data) == 0:
            return self._majority_class(parent_data if len(data) == 0 else data)
        else:
            info_gains = {feature: self._calc_info_gain(data, feature) for feature in features}
            best_feature = max(info_gains, key=info_gains.get)

            tree = {best_feature: {}}
            remaining_features = [f for f in features if f != best_feature]

            for value in data[best_feature].unique():
                subset = data[data[best_feature] == value]
                tree[best_feature][value] = self._id3(subset, remaining_features, data)

            return tree

    @staticmethod
    def _calc_entropy(y: pd.Series) -> float:
        """
        计算信息熵

        Args:
            y (pd.Series): 输入标签。

        Returns:
            信息熵 (float)
        """
        probs = y.value_counts(normalize=True)
        entropy = -np.sum([p * np.log2(p) for p in probs])
        return entropy

    def _calc_info_gain(self, data: pd.DataFrame, feature: str) -> float:
        """
        计算信息增益

        Args:
            data (pd.DataFrame): 输入样本。
            feature (str): 当前特征。

        Returns:
            信息增益 (float)
        """
        original_entropy = self._calc_entropy(data[self.target])
        weighted_entropy = np.sum([
            (len(subset) / len(data)) * self._calc_entropy(subset[self.target])
            for value, subset in data.groupby(feature)
        ])
        return original_entropy - weighted_entropy

    def _majority_class(self, data: pd.DataFrame) -> str:
        """
        获取数据中数量最多的类别。

        Args:
            data (pd.DataFrame): 输入样本。

        Returns:
            数量最多的类别 (str)
        """
        return data[self.target].value_counts().idxmax()


if __name__ == '__main__':
    # Example usage:
    data = pd.DataFrame({
        'x1': [1, 1, 1, 1, 2, 2, 2, 2],
        'x2': [1, 1, 2, 2, 1, 1, 2, 2],
        'x3': [1, 2, 1, 2, 1, 2, 1, 2],
        'y': [3, 1, 3, 2, 3, 2, 3, 3]
    })

    features = ['x1', 'x2', 'x3']
    target = 'y'
