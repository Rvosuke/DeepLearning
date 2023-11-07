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


def train(epoches, model, optimizer, loss_fn, x, y):
    model.train()
    for epoch in range(1, 1 + epoches):
        optimizer.zero_grad()
        output = model(x)
        loss = loss_fn(output, y)
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
    learning_rate = 1e-3
    fcn = FCN()
    opt = optim.Adam(fcn.parameters(), lr=learning_rate)
    ce = nn.CrossEntropyLoss()
    x_train, x_test, y_train, y_test = iris_data_load()
    x_train = torch.from_numpy(x_train).float()
    x_test = torch.from_numpy(x_test).float()
    y_train = torch.from_numpy(y_train).long()
    y_test = torch.from_numpy(y_test).long()

    train(1000, fcn, opt, ce, x_train, y_train)
    test(fcn, x_test, y_test)


if __name__ == '__main__':
    main()
