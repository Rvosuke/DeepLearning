from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def create_dataset(data_dir, training=True, batch_size=64,
                   resize=(32, 32), rescale=1 / (255 * 0.3081), shift=-0.1307 / 0.3081):
    # 这里的shift和rescale是根据MNIST数据集的均值和标准差提前计算得到的
    # 数据处理：数据集加载、缩放、归一化、格式转换、洗牌、批标准化。
    transform = transforms.Compose([
        transforms.Resize(resize),
        transforms.ToTensor(),
        transforms.Normalize(mean=[shift], std=[rescale])
    ])

    dataset = datasets.MNIST(root=data_dir, train=training, transform=transform, download=True)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=training)

    return loader
