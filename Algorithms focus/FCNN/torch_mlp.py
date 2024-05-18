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


def train(epoches, model, optimizer, loss_fn, data_loader, device, accu_step=3):
    model.train()
    # step_lr = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.9)
    for epoch in range(1, 1 + epoches):
        for index, (X, Y) in enumerate(data_loader):
            X, Y = X.to(device), Y.to(device)
            output = model(X)
            loss = loss_fn(output, Y)

            loss.backward()
            # step_lr.step()
            if epoch % accu_step == 0:
                optimizer.step()
                optimizer.zero_grad()
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
    learning_rate = 2e-2
    fcn = FCN().to(device)
    opt = optim.Adam(fcn.parameters(), lr=learning_rate)
    ce = nn.CrossEntropyLoss().to(device)
    x_train, x_test, y_train, y_test = iris_data_load()
    x_train = torch.from_numpy(x_train).to(device)
    x_test = torch.from_numpy(x_test).to(device)
    y_train = torch.from_numpy(y_train).long().to(device)
    y_test = torch.from_numpy(y_test).long().to(device)
    train_data = torch.utils.data.TensorDataset(x_train, y_train)
    train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=128, shuffle=True)
    # 设置为GPU模式

    train(128, fcn, opt, ce, train_loader, device=device)
    test(fcn, x_test, y_test)
    n_params = 0
    for param in fcn.parameters():
        n_params += torch.numel(param)
    print('Number of Parameters: {}'.format(n_params))


if __name__ == '__main__':
    import time
    start = time.time()
    main()
    end = time.time()
    print('Time: {:.2f}s'.format(end - start))
