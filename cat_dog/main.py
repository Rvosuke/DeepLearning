import time

import torch
from torch.utils.data import DataLoader
from cat_dog_model import CatDogModel
from cat_dog_dataset import CatDogDataset
from cat_dog_train import train, test


def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    train_dataset = CatDogDataset(train=True)
    test_dataset = CatDogDataset(train=False)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=8)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True, num_workers=8)

    model = CatDogModel().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)
    loss_fn = torch.nn.CrossEntropyLoss()
    train(model, 10, train_loader, optimizer, loss_fn, device)
    test(model, test_loader, device)
    torch.save(model, f'{time.time()}cat_dog_model.pth')
    print('model saved')


if __name__ == '__main__':
    main()
