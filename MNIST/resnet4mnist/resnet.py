import json
import torch
import logging
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch import nn, optim
from torch.optim.lr_scheduler import StepLR
from datetime import datetime
from MNIST.utils.view import plot_history
from MNIST.utils.mnist_dataload import mnist_dataset
from MNIST.utils.device_set import device_setting
from MNIST.utils.log import setup_logger

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")  # 记录当前时间


class ResBlock(nn.Module):
    def __init__(self, n_chans):
        super(ResBlock, self).__init__()
        self.conv = nn.Conv2d(n_chans, n_chans, kernel_size=3, padding=1, bias=False)
        self.batch_norm = nn.BatchNorm2d(num_features=n_chans)
        # 批范数层会抵消偏置的影响, 因此它通常被排除在外
        torch.nn.init.kaiming_normal_(self.conv.weight, nonlinearity='relu')
        torch.nn.init.constant_(self.batch_norm.weight, 0.5)
        torch.nn.init.zeros_(self.batch_norm.bias)

    def forward(self, x):
        out = self.conv(x)
        out = self.batch_norm(out)
        out = torch.relu(out)
        return out + x


class ResNet(nn.Module):
    def __init__(self, n_chans1=32, n_blocks=16):
        super().__init__()
        self.n_chans1 = n_chans1  # 通道数
        self.conv1 = nn.Conv2d(1, n_chans1, kernel_size=3, padding=1)
        self.resblocks = nn.Sequential(*(n_blocks * [ResBlock(n_chans=n_chans1)]))
        self.fc1 = nn.Linear(7 * 7 * n_chans1, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        out = F.max_pool2d(torch.relu(self.conv1(x)), 2)  # nc*28*28 -> nc*14*14
        out = self.resblocks(out)  # nc*14*14 -> nc*14*14
        out = F.max_pool2d(out, 2)  # nc*14*14 -> nc*7*7
        out = out.view(x.shape[0], -1)
        out = torch.relu(self.fc1(out))
        out = self.fc2(out)
        return out


def train(batch_size=128, lr=0.001, num_epochs=60):
    ds_train, ds_eval = mnist_dataset(batch_size=batch_size)
    device = device_setting()
    net = ResNet()
    loss_fn = nn.CrossEntropyLoss()
    opt = optim.Adam(net.parameters(), lr=lr)
    net.to(device)

    train_losses = []
    val_accuracies = []

    for epoch in range(1, 1 + num_epochs):
        running_loss = 0.0
        batch_num = 1
        for i, (inputs, labels) in enumerate(ds_train, 0):
            inputs, labels = inputs.to(device), labels.to(device)  # 将输入和标签移动到所选设备
            opt.zero_grad()
            outputs = net(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            opt.step()
            running_loss += loss.item()
            batch_num = i
        batch_num += 1
        running_loss /= batch_num
        train_losses.append(running_loss)
        logging.info(f'Epoch {epoch}, Loss: {running_loss:.4f}')

        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in ds_eval:
                images, labels = images.to(device), labels.to(device)
                outputs = net(images)
                predicted = torch.argmax(outputs.data, 1)
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
        'batch_size': 256,
        'lr': 5e-5,
        'num_epochs': 40,
        'model': 'ResNet16'
    }

    # 记录训练的开始时间
    logging.info(f"Training with: \n{hyperparameters}\n")

    # 训练模型
    net, train_losses, val_acc = (
        train(hyperparameters['batch_size'], hyperparameters['lr'], hyperparameters['num_epochs']))

    # 绘制训练历史记录
    title = f"ResNet_{timestamp}(lr={hyperparameters['lr']}, epochs={hyperparameters['num_epochs']})"
    plot_history(train_losses, val_acc, title)

    # 将超参数保存到 JSON
    with open(f'resnet.hyperparameters.json', 'w') as f:
        json.dump(hyperparameters, f, indent=4)

    # 保存模型参数
    torch.save(net.state_dict(), f'resnet_{timestamp}_emnist.pth')


if __name__ == '__main__':
    main()
