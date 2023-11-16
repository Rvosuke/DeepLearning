from torch import nn


class Generator(nn.Module):
    def __init__(self, h, w, noise_len=100):
        super(Generator, self).__init__()
        self.h = h
        self.w = w
        self.fc1 = nn.Linear(noise_len, 2048)
        self.fc2 = nn.Linear(2048, 1024)
        self.fc3 = nn.Linear(1024, self.h * self.w)

    def forward(self, x):
        out = self.fc1(x)
        out = nn.ReLU(True)(out)
        out = self.fc2(out)
        out = nn.ReLU(True)(out)
        out = self.fc3(out)
        out.reshape(x.shape[0], 1, self.h, self.w)
        return out


class Discriminator(nn.Module):
    def __init__(self, h, w):
        super(Discriminator, self).__init__()
        self.h = h
        self.w = w
        self.fc1 = nn.Linear(self.h * self.w, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x):
        x.reshape(x.shape[0], -1)
        out = self.fc1(x)
        out = nn.ReLU(True)(out)
        out = self.fc2(out)
        out = nn.ReLU(True)(out)
        out = self.fc3(out)
        return out
