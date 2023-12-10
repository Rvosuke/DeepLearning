import numpy as np


def softmax(x):
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / np.sum(e_x, axis=1, keepdims=True)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def cross_entropy_loss(predictions, labels):
    m = labels.shape[0]
    log_likelihood = -np.log(predictions[range(m), labels])
    loss = np.sum(log_likelihood) / m
    return loss


def derivative_cross_entropy(predictions, labels):
    m = labels.shape[0]
    grad = predictions.copy()
    grad[range(m), labels] -= 1
    grad = grad / m
    return grad


def tanh(x):
    # 手动实现tanh函数
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))


def derivative_tanh(x):
    return 1 - tanh(x) ** 2


# 小批量随机梯度下降的数据打包函数
def batch_generator(X, y, batch_size=32):
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    for start_idx in range(0, X.shape[0] - batch_size + 1, batch_size):
        excerpt = indices[start_idx:start_idx + batch_size]
        yield X[excerpt], y[excerpt]


# MLP类
class MLP:
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        self.W1 = np.random.randn(input_size, hidden_size1) / np.sqrt(input_size)
        self.b1 = np.zeros(hidden_size1)
        self.W2 = np.random.randn(hidden_size1, hidden_size2) / np.sqrt(hidden_size1)
        self.b2 = np.zeros(hidden_size2)
        self.W3 = np.random.randn(hidden_size2, output_size) / np.sqrt(hidden_size2)
        self.b3 = np.zeros(output_size)

    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = tanh(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = tanh(self.z2)
        self.z3 = np.dot(self.a2, self.W3) + self.b3
        self.a3 = softmax(self.z3)
        return self.a3


# 对于MLP，我们需要一个训练函数来执行前向和后向传播
def train(model, X_train, y_train, X_val, y_val, epochs, batch_size, learning_rate):
    n_batches = int(np.ceil(X_train.shape[0] / batch_size))

    for epoch in range(epochs):
        for X_batch, y_batch in batch_generator(X_train, y_train, batch_size):
            # 前向传播
            output = model.forward(X_batch)

            # 计算损失
            loss = cross_entropy_loss(output, y_batch)

            # 反向传播
            dZ3 = derivative_cross_entropy(output, y_batch)
            dW3 = np.dot(model.a2.T, dZ3)
            db3 = np.sum(dZ3, axis=0, keepdims=True)

            dA2 = np.dot(dZ3, model.W3.T)
            dZ2 = dA2 * derivative_tanh(model.z2)
            dW2 = np.dot(model.a1.T, dZ2)
            db2 = np.sum(dZ2, axis=0, keepdims=True)

            dA1 = np.dot(dZ2, model.W2.T)
            dZ1 = dA1 * derivative_tanh(model.z1)
            dW1 = np.dot(X_batch.T, dZ1)
            db1 = np.sum(dZ1, axis=0, keepdims=True)

            # 更新权重
            model.W3 -= learning_rate * dW3
            model.b3 -= learning_rate * db3.squeeze()
            model.W2 -= learning_rate * dW2
            model.b2 -= learning_rate * db2.squeeze()
            model.W1 -= learning_rate * dW1
            model.b1 -= learning_rate * db1.squeeze()

        # 这里添加验证逻辑，以监控验证损失和准确性
        val_output = model.forward(X_val)
        val_loss = cross_entropy_loss(val_output, y_val)
        val_predictions = np.argmax(val_output, axis=1)
        val_accuracy = np.mean(val_predictions == y_val)
        print(
            f'Epoch {epoch + 1}, Loss: {loss:.4f}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}')


if __name__ == '__main__':
    from iris_data import iris_data_load

    # 加载数据
    X_train_scaled, X_test_scaled, y_train, y_test = iris_data_load()
    # 训练模型
    input_size = X_train_scaled.shape[1]
    hidden_size1 = 10
    hidden_size2 = 10
    output_size = len(np.unique(y_train))  # 根据类别数量来设置
    model = MLP(input_size, hidden_size1, hidden_size2, output_size)

    # 转换标签为one-hot编码
    y_train_one_hot = np.eye(output_size)[y_train]
    y_test_one_hot = np.eye(output_size)[y_test]

    train(model, X_train_scaled, y_train, X_test_scaled, y_test, epochs=1000, batch_size=32, learning_rate=1e-2)

    # 最后，使用测试集评估模型性能
    test_output = model.forward(X_test_scaled)
    test_loss = cross_entropy_loss(test_output, y_test)
    test_predictions = np.argmax(test_output, axis=1)
    test_accuracy = np.mean(test_predictions == y_test)
    print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}')
