import time

import torch


def train(model, epochs, data_loader, optimizer, loss_fn, device):
    model = model.to(device)
    model.train()
    start_time = time.time()
    for epoch in range(1, 1+epochs):
        epoch_time = time.time()
        loss_list = []
        for index, data in enumerate(data_loader):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, labels.long())
            loss.backward()
            loss_list.append(loss)
            optimizer.step()
        loss = sum(loss_list) / len(loss_list)
        epoch_time = time.time() - epoch_time

        print(f"Epoch: {epoch}, Loss: {loss:.4f}, Time: {int(epoch_time)}s")
    end_time = time.time() - start_time
    print(f"Total Time: {int(end_time)}s")


def test(model, dataloader, device):
    model.eval()
    correct = 0
    with torch.no_grad():
        for index, data in enumerate(dataloader):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            predicted = torch.argmax(outputs.data, 1)
            correct += (predicted == labels).long().sum()
    print(f"Accuracy: {correct / len(dataloader.dataset):.2f}")
