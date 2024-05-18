import torch
from torch.utils.data import TensorDataset, DataLoader


def split_data(X: torch.Tensor, Y: torch.Tensor, test_rate: float = 0.2, shuffle=False) -> tuple:
    """
    将数据集分割为训练集和测试集。

    参数:
    X -- 特征数据集 (torch.Tensor)
    Y -- 目标数据集 (torch.Tensor)
    test_rate -- 测试集占总数据集的比例 (float, 默认值: 0.2)

    返回值:
    train -- 训练集特征 (torch.Tensor)
    train_label -- 训练集标签 (torch.Tensor)
    test -- 测试集特征 (torch.Tensor)
    test_label -- 测试集标签 (torch.Tensor)
    """

    n = len(Y)
    if shuffle:
        perm = torch.randperm(n)
        X, Y = X[perm], Y[perm]
    test_size = int(n * test_rate)

    test, test_label = X[:test_size], Y[:test_size]
    train, train_label = X[test_size:], Y[test_size:]

    return train, train_label, test, test_label


def batch_generator(X: torch.Tensor, Y: torch.Tensor, batch_size: int) -> tuple:
    """
    生成批次数据。

    参数:
    X -- 特征数据集 (torch.Tensor)
    Y -- 目标数据集 (torch.Tensor)
    batch_size -- 批次大小 (int)

    返回值:
    X_batch -- 批次特征 (torch.Tensor)
    Y_batch -- 批次标签 (torch.Tensor)
    """

    n = len(Y)
    data_loader = []

    for i in range(0, n, batch_size):
        X_batch, Y_batch = X[i:i + batch_size], Y[i:i + batch_size]
        data_loader.append((X_batch, Y_batch))
    return tuple(data_loader)


if __name__ == '__main__':
    trainX, trainY = torch.randn(1428, 3), torch.randn(1428)
    train_set = TensorDataset(trainX, trainY)
    train_loader = DataLoader(dataset=train_set,  # 调用打包函数
                              batch_size=100,  # 包的大小
                              shuffle=True)  # 默认shuffle=False
