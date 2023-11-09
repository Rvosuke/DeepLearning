import torch
from torch import nn, optim
from iris_data import iris_data_load


class FCN(nn.Module):
    def __init__(self, input_size=4, hidden_size=10, output_size=3):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = torch.tanh(out)
        out = self.fc2(out)
        out = torch.tanh(out)
        out = self.fc3(out)
        return out


def train(epoches, model, optimizer, loss_fn, data_loader):
    model.train()
    for epoch in range(1, 1 + epoches):
        for index, data in enumerate(data_loader):
            optimizer.zero_grad()
            output = model(data[0])
            loss = loss_fn(output, data[1])
            loss.backward()
            optimizer.step()
        if epoch % 10 == 0:
            print('Epoch {}, Loss {}'.format(epoch, loss.item()))


def test(model, x, y):
    model.eval()
    with torch.no_grad():
        output = model(x)
        _, predicted = torch.max(output.data, 1)
        correct = (predicted == y).sum().item()
        print('Accuracy: {}/{} ({:.0f}%)'.format(correct, len(y), 100. * correct / len(y)))


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    learning_rate = 1e-3
    fcn = FCN().to(device)
    opt = optim.Adam(fcn.parameters(), lr=learning_rate)
    ce = nn.CrossEntropyLoss().to(device)
    x_train, x_test, y_train, y_test = iris_data_load()
    x_train = torch.from_numpy(x_train).to(device)
    x_test = torch.from_numpy(x_test).to(device)
    y_train = torch.from_numpy(y_train).long().to(device)
    y_test = torch.from_numpy(y_test).long().to(device)
    train_data = torch.utils.data.TensorDataset(x_train, y_train)
    train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=8, shuffle=True, num_workers=8)
    # 设置为GPU模式

    train(1000, fcn, opt, ce, train_loader)
    test(fcn, x_test, y_test)


if __name__ == '__main__':
    main()
