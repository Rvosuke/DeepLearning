import json
import torch
import logging
from torch import nn, optim
from datetime import datetime
from MNIST.utils.view import plot_history
from MNIST.utils.mnist_dataload import create_dataset
from MNIST.utils.device_set import device_setting
from MNIST.utils.log import setup_logger

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


class LeNet5(nn.Module):
    """模型定义：算子初始化（参数设置），网络构建。"""

    def __init__(self, activation='relu', dropout_rate=None):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, stride=1, padding=0)
        self.conv2 = nn.Conv2d(6, 16, 5, stride=1, padding=0)
        self.activation = activation
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout_rate = dropout_rate
        self.dropout = nn.Dropout(p=dropout_rate) if dropout_rate else None
        self.fc1 = nn.Linear(400, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        act = self.relu if self.activation == 'relu' else self.sigmoid
        x = act(self.conv1(x))
        x = self.pool(x)
        x = act(self.conv2(x))
        x = self.pool(x)

        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc1(x)
        if self.dropout:
            x = self.dropout(x)
        x = self.fc2(x)
        if self.dropout:
            x = self.dropout(x)
        x = self.fc3(x)
        x = self.softmax(x)

        return x


def train(data_dir, loss_type='ce', activation='relu', dropout_rate=0.1, lr=0.01, momentum=0.9, num_epochs=10):
    ds_train = create_dataset(data_dir, training=True)
    ds_eval = create_dataset(data_dir, training=False)

    net = LeNet5(activation=activation, dropout_rate=dropout_rate)
    device = device_setting()
    net.to(device)
    loss = nn.CrossEntropyLoss() if loss_type == 'ce' else nn.NLLLoss()

    opt = optim.SGD(net.parameters(), lr=lr, momentum=momentum)

    train_losses = []
    val_accuracies = []

    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, data in enumerate(ds_train, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)  # 将输入和标签移动到所选设备
            opt.zero_grad()
            outputs = net(inputs)
            loss_output = loss(outputs, labels)
            loss_output.backward()
            opt.step()

            running_loss += loss_output.item()
        train_losses.append(running_loss)

        logging.info(f'Epoch {epoch + 1}, Loss: {running_loss:.4f}')

        # 每个epoch结束后，计算在验证集上的准确率
        correct = 0
        total = 0
        with torch.no_grad():
            for data in ds_eval:
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs = net(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        acc = 100 * correct / total
        val_accuracies.append(acc)
        logging.info(f'Accuracy: {acc:.2f} %')
    return net, train_losses, val_accuracies


def main():
    setup_logger(timestamp)

    # 定义超参数
    hyperparameters = {
        'data_dir': '../MNIST_Data',
        'activation': 'relu',
        'loss_type': 'ce',
        'num_epochs': 128,
        'lr': 1e-4,
        'momentum': 0.9,
        'dropout_rate': 0.4,
    }

    # 记录训练的开始时间
    logging.info(f"Training with hyperparameters: \n{hyperparameters}\n")

    # 训练模型
    net, train_losses, val_acc = train(**hyperparameters)

    # 绘制训练历史记录
    title = f"LeNet_{timestamp}(lr={hyperparameters['lr']}, epochs={hyperparameters['num_epochs']})"
    plot_history(train_losses, val_acc, title)

    # 保存模型参数
    torch.save(net.state_dict(), f"LeNet_{timestamp}.pth")


if __name__ == '__main__':
    main()
