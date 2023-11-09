import os
import torch
from torchvision import datasets, transforms


def get_file_label(tmp_path):
    """
    读取每个文件的相对路径（含文件名）及其类别（分别用 0、1、2、3、4 对类别 编号），形成以二元组(路径, 类别编号)为元素的列表。

    :param tmp_path:
    :return:
    """
    file_label = []
    for root, dirs, files in os.walk(tmp_path):
        for file in files:
            if file.endswith('.jpg'):
                file_path = os.path.join(root, file)
                label = root.split('/')[-1]
                if label == 'daisy':
                    file_label.append((file_path, 0))
                elif label == 'dandelion':
                    file_label.append((file_path, 1))
                elif label == 'rose':
                    file_label.append((file_path, 2))
                elif label == 'sunflower':
                    file_label.append((file_path, 3))
                elif label == 'tulip':
                    file_label.append((file_path, 4))
    return file_label


class FlowerDataSet(Dataset):
    def __init__(self, file_label, train, transform=None):
        self.file_label = file_label
        self.transform = transform
        self.train = train

    def train_test_split(self, test_size=0.2, shuffle=True):
        """
        划分训练集和测试集

        :param test_size:
        :param shuffle:
        :return:
        """
        length = len(self.file_label)
        if shuffle:
            indices = torch.randperm(length)
        else:
            indices = torch.arange(length)
        test_length = int(length * test_size)
        train_length = length - test_length
        train_indices = indices[:train_length]
        test_indices = indices[train_length:]
        train_file_label = [self.file_label[i] for i in train_indices]
        test_file_label = [self.file_label[i] for i in test_indices]
        train_dataset = FlowerDataSet(train_file_label, train=True, transform=self.transform)
        test_dataset = FlowerDataSet(test_file_label, train=False, transform=self.transform)
        return train_dataset, test_dataset

    def __getitem__(self, index):

        file_path, label = self.file_label[index]
        img = Image.open(file_path)
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        length = len(self.file_label)
        return length