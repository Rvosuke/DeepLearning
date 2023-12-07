import numpy as np
import re
from collections import Counter
from scipy import stats


class NaiveBayes:
    def __init__(self, filepath):
        self.filepath = filepath
        self.text = []
        self.text_len = 0
        self.attributes_len = 0
        self.class_count = None
        self.prior_p = None
        self.class_name = None
        self.attributes_count_list = None
        self.guess_num = 0
        self.guess_index = set()
        self.guess_means = None
        self.guess_stds = None

    def load_data(self):
        with open(self.filepath, mode='r', encoding='utf-8') as f:
            # Skip attribute labels
            next(f)
            self.text = [line.strip().split(',')[1:] for line in f]
        self.text_len = len(self.text)
        self.attributes_len = len(self.text[0]) - 1

    @staticmethod
    def is_float(s):
        """判断字符串s是否为浮点数"""
        pattern = re.compile(r'^[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?$')
        return bool(pattern.match(s))

    def compute_prior_probabilities(self):
        self.class_count = Counter(row[-1] for row in self.text)
        self.class_name = list(self.class_count.keys())
        self.prior_p = np.array([num / self.text_len for num in self.class_count.values()], dtype=float)

    def compute_class_condition_probabilities(self):
        self.attributes_count_list = [[Counter() for _ in range(self.attributes_len)] for _ in range(len(self.class_name))]
        for row in self.text:
            for i, val in enumerate(row[:-1]):
                if not self.is_float(val):
                    for j, class_label in enumerate(self.class_name):
                        if row[-1] == class_label:
                            self.attributes_count_list[j][i][val] += 1
                else:
                    self.guess_num += 1
                    self.guess_index.add(i)

    def compute_continuous_variables(self):
        self.guess_means = np.zeros((len(self.class_name), self.guess_num), dtype=float)
        self.guess_stds = np.zeros((len(self.class_name), self.guess_num), dtype=float)
        for i, class_label in enumerate(self.class_name):
            guess_vector = np.array([[float(val) for j, val in enumerate(row[:-1]) if self.is_float(val) and row[-1] == class_label] for row in self.text]).T
            self.guess_means[i, :] = np.mean(guess_vector, axis=1)  # 计算每行的均值
            self.guess_stds[i, :] = np.std(guess_vector, axis=1)  # 计算每行的标准差

    def classify(self, test):
        condition_p = np.zeros((len(self.class_name), self.attributes_len), dtype=float)
        for i, val in enumerate(test[:-1]):
            if not self.is_float(val):
                class_condition_p = [self.attributes_count_list[k][i][val] / self.class_count[self.class_name[k]] for k in range(len(self.class_name))]
            else:
                j = list(self.guess_index).index(i)
                class_condition_p = [stats.norm.pdf(float(val), self.guess_means[k][j], self.guess_stds[k][j]) for k in range(len(self.class_name))]
            condition_p[:, i] = class_condition_p

        class_g = [self.prior_p[i] * np.prod(condition_p[i]) for i in range(len(self.class_name))]
        return self.class_name[np.argmax(class_g)]

    def fit(self):
        self.load_data()
        self.compute_prior_probabilities()
        self.compute_class_condition_probabilities()
        self.compute_continuous_variables()


if __name__ == "__main__":
    classifier = NaiveBayes('../../res/data_3.0.txt')
    classifier.fit()
    test = ['青绿', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', '0.697', '0.460', '?']
    decision = classifier.classify(test)
    print(decision)
