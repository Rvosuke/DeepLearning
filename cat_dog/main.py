import time

import torch
from torch.utils.data import DataLoader
from cat_dog_model import CatDogBasicModel, CatDogVgg16, CatDogResNet50
from cat_dog_dataset import CatDogDataset
from cat_dog_train import train, test


def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    train_dataset = CatDogDataset(train=True)
    test_dataset = CatDogDataset(train=False)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)

    model = CatDogResNet50().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=3e-3, momentum=0.9)
    loss_fn = torch.nn.CrossEntropyLoss()
    model = train(model, 20, train_loader, optimizer, loss_fn, device)
    acc = test(model, test_loader, device)
    torch.save(model, f'cat_dog_model_{int(acc)}.pth')
    print('model saved')


if __name__ == '__main__':
    main()
