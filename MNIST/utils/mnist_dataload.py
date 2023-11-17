from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def mnist_dataset(batch_size=128):
    # 这里的shift和rescale是根据MNIST数据集的均值和标准差提前计算得到的
    # 数据处理：数据集加载、缩放、归一化、格式转换、洗牌、批标准化。
    rescale = 1 / (255 * 0.3081)
    shift = -0.1307 / 0.3081
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[shift], std=[rescale])
    ])

    dataset = datasets.MNIST(root="../MNIST_Data", train=True, transform=transform, download=True)
    test = datasets.MNIST(root="../MNIST_Data", train=False, transform=transform, download=True)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test, batch_size=batch_size*2)

    return loader, test_loader


def emnist_dataset(batch_size=64, transform=transforms.ToTensor()):
    t = transforms.Compose([
        transform
    ])

    train_dataset = datasets.EMNIST(root="../MNIST_Data", split='letters', train=True, transform=t, download=True)
    test_dataset = datasets.EMNIST(root="../MNIST_Data", split='letters', train=False, transform=t, download=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    return train_loader, test_loader
