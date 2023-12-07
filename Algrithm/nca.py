import numpy as np
import torch
from time import time
from typing import Callable, Tuple


def timing(func: Callable[..., torch.Tensor]) -> Callable[..., torch.Tensor]:
    """Decorator for timing functions."""

    def wrapper(*args, **kwargs):
        start = time()
        result = func(*args, **kwargs)
        end = time()
        print(f"Running {func.__name__} took {end - start:.2f} seconds.")
        return result

    return wrapper


class NCA:
    def __init__(self, dim: int, device: str = 'cuda'):
        self.dim = dim
        self.device = device
        self.W = torch.randn(self.dim, self.dim, device=self.device, requires_grad=True)

    @timing
    def fit(self, X: torch.Tensor, y: torch.Tensor, lr: float = 0.01, epochs: int = 100):
        optimizer = torch.optim.SGD([self.W], lr=lr)

        for _ in range(epochs):
            optimizer.zero_grad()

            diff = X[None, :, :] - X[:, None, :]
            dist = torch.sum((diff @ self.W.T) ** 2, dim=-1)
            exp_dist = torch.exp(-dist)
            p_ij = exp_dist / (torch.sum(exp_dist, dim=-1, keepdim=True) - torch.diag(exp_dist))

            p_i = torch.sum(p_ij, dim=-1)
            p = p_ij[y == y[:, None]]

            loss = -torch.sum(p * torch.log(p_i))

            loss.backward()
            optimizer.step()

    def transform(self, X: torch.Tensor) -> torch.Tensor:
        return X @ self.W


if __name__ == '__main__':
    from torchvision import datasets, transforms

    # 定义数据预处理方式
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # 从torchvision中下载并加载MNIST数据集
    mnist_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

    # 取出数据和标签
    X = mnist_data.data.float().view(-1, 28 * 28).to('cuda')
    y = mnist_data.targets.to('cuda')

    # 定义并训练NCA模型
    nca = NCA(dim=2)
    nca.fit(X, y, lr=0.01, epochs=100)

    # 使用训练好的模型进行数据转换
    X_transformed = nca.transform(X)

    # 将GPU上的数据移到CPU上，并转换为numpy数组
    X_transformed = X_transformed.cpu().numpy()
