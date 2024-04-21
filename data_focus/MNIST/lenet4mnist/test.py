# 导入Lenet参数模型，并进行测试推理
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.autograd import Variable

# 定义是否使用GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 加载模型
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


model = LeNet5()
model.load_state_dict(torch.load('model.pth'))
model.to(device)
model.eval()

# 定义对数据的预处理
data_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor()
])

# 加载测试数据
test_dataset = datasets.MNIST(
    root='./data', train=False, download=True, transform=data_transform)

test_loader = torch.utils.data.DataLoader(
    dataset=test_dataset,
    batch_size=1,
    shuffle=False
)

# 开始测试
correct = 0
total = 0
