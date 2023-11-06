import json
import torch
import logging
import matplotlib.pyplot as plt
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms
from datetime import datetime


# 数据处理：数据集加载、缩放、归一化、格式转换、洗牌、批标准化。
def create_dataset(data_dir, training=True, batch_size=32,
                   resize=(32, 32), rescale=1 / (255 * 0.3081), shift=-0.1307 / 0.3081):
    # 这里的shift和rescale是根据MNIST数据集的均值和标准差提前计算得到的
    transform = transforms.Compose([
        transforms.Resize(resize),
        transforms.ToTensor(),
        transforms.Normalize(mean=[shift], std=[rescale])
    ])

    ds = MNIST(root=data_dir, train=training, transform=transform, download=True)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=training, num_workers=4, pin_memory=True, drop_last=True)

    return loader


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


def device_setting(device_type='cuda', device_id=0, num_CPUs=8):
    """设备设置：CPU/GPU、单卡/多卡、多线程。"""
    # 设置使用的设备
    device = torch.device(f"{device_type}:{device_id}" if torch.cuda.is_available() else "cpu")
    if device.type == 'cuda':
        torch.cuda.set_device(device_id)  # 设置使用的 GPU
        torch.backends.cudnn.benchmark = True  # 自动寻找最快的卷积算法
    else:
        torch.set_num_threads(num_CPUs)  # 设置PyTorch线程数
    return device


def train(data_dir, loss_type='ce', activation='relu', dropout_rate=0.1, lr=0.01, momentum=0.9, num_epochs=10):
    ds_train = create_dataset(data_dir, training=True)
    ds_eval = create_dataset(data_dir, training=False)

    net = LeNet5(activation=activation, dropout_rate=dropout_rate)
    device = device_setting()
    net.to(device)

    if loss_type == 'mse':
        loss = nn.MSELoss()
    elif loss_type == 'ce':
        loss = nn.CrossEntropyLoss()
    else:
        raise ValueError("Invalid loss_type. Choose either 'mse' or 'ce'")

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
            if loss_type == 'mse':
                labels = labels.long()
                labels_one_hot = torch.zeros(labels.shape[0], 10).to(device)
                labels_one_hot.scatter_(1, labels.view(-1, 1), 1)
                loss_output = loss(outputs, labels_one_hot)
            else:
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


def plot_history(train_losses, val_acc, title):
    epochs = range(1, len(train_losses) + 1)

    plt.figure()
    plt.plot(epochs, train_losses, 'r', label='Training loss')
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.figure()
    plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.show()


def setup_logger():
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_filename = f"training_logs_{timestamp}.log"

    logging.basicConfig(filename=log_filename,
                        filemode='w',
                        level=logging.INFO,
                        format='%(asctime)s [%(levelname)s] - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] - %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)


def main():
    setup_logger()

    # 定义超参数
    hyperparameters = {
        'data_dir': 'MNIST_Data',
        'activation': 'relu',
        'loss_type': 'ce',
        'num_epochs': 20,
        'lr': 0.001,
        'momentum': 0.9,
        'dropout_rate': 0.1,
    }

    # 记录训练的开始时间
    logging.info(f"Training with hyperparameters: {hyperparameters}")

    # 训练模型
    net, train_losses, val_acc = train(**hyperparameters)

    # 绘制训练历史记录
    title = f"Train with {hyperparameters['activation']} {hyperparameters['loss_type']} with Dropout {hyperparameters['dropout_rate']}"
    plot_history(train_losses, val_acc, title)

    # 将超参数保存到 JSON
    with open('hyperparameters.json', 'w') as f:
        json.dump(hyperparameters, f, indent=4)

    # 保存模型参数
    model = net.state_dict()  # 获取模型参数的实际方法的占位符
    with open('model_parameters.json', 'w') as f:
        json.dump(model, f, indent=4)


if __name__ == '__main__':
    main()
