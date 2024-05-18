import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 加载数据集
digits = load_digits()
X, y = digits.data, digits.target

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# 定义 Sigmoid 激活函数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# 定义神经网络类
class ThresholdFullyConnectedLayer:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        self.W1 = np.random.randn(input_size, hidden_size)
        self.W2 = np.random.randn(hidden_size, output_size)
        self.threshold = np.random.randn(hidden_size)
        self.a1 = np.zeros(hidden_size)
        self.learning_rate = learning_rate

    def forward(self, X):
        self.z1 = np.dot(X, self.W1)
        self.a1 = sigmoid(self.z1)
        self.a1[self.z1 < self.threshold] = 0  # 应用阈值
        self.z2 = np.dot(self.a1, self.W2)
        self.a2 = sigmoid(self.z2)
        return self.a2

    def backward(self, X, y, output):
        # 反向传播,计算梯度
        m = X.shape[0]
        output_error = output - y
        output_delta = output_error * (output * (1 - output))

        z1_error = np.dot(output_delta, self.W2.T)
        z1_delta = z1_error * (self.a1 * (1 - self.a1))

        self.W2 -= self.learning_rate * np.dot(self.a1.T, output_delta) / m
        self.W1 -= self.learning_rate * np.dot(X.T, z1_delta) / m

    def train(self, X, y, epochs):
        for i in range(epochs):
            output = self.forward(X)
            self.backward(X, y, output)
            self.evolve_threshold(output)  # 进化阈值

    def evolve_threshold(self, output, pop_size=50, max_iter=100, F=0.5, CR=0.7):
        # 初始化种群
        population = [np.random.randn(self.threshold.shape[0]) for _ in range(pop_size)]
        best_score = float('inf')
        best_individual = None

        for _ in range(max_iter):
            for i in range(pop_size):
                # 随机选择三个其他个体
                indices = [j for j in range(pop_size) if j != i]
                a, b, c = population[np.random.choice(indices, 3, replace=False)]

                # 计算差分向量并生成试验向量
                mutant = a + F * (b - c)
                # 交叉操作
                cross_points = np.random.rand(self.threshold.shape[0]) < CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.threshold.shape[0])] = True
                trial = np.where(cross_points, mutant, population[i])

                # 计算适应度
                self.threshold = trial
                trial_score = self.compute_fitness(output)

                # 选择操作
                if trial_score < best_score:
                    best_score = trial_score
                    best_individual = trial

            # 更新种群
            population[i] = best_individual

        # 更新阈值
        self.threshold = best_individual

    def dropout_layer(X, dropout_rate):
        assert 0 <= dropout_rate <= 1
        # 在dropout_rate为0时，所有元素都被保留
        if dropout_rate == 0:
            return X
        mask = np.random.uniform(0, 1, X.shape) > dropout_rate
        return mask * X / (1.0 - dropout_rate)


# 训练模型
model = ThresholdFullyConnectedLayer(input_size=64, hidden_size=32, output_size=10)
model.train(X_train, y_train, epochs=100)

# 测试模型
y_pred = model.forward(X_test)
y_pred = np.argmax(y_pred, axis=1)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
