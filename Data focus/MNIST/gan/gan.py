import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from typing import Tuple


# 定义生成器
class Generator(nn.Module):
    def __init__(self, latent_dim: int, img_shape: Tuple[int, int, int]):
        super(Generator, self).__init__()
        self.img_shape = img_shape

        def block(in_features: int, out_features: int, normalize: bool = True):
            layers = [nn.Linear(in_features, out_features)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_features, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(torch.prod(torch.tensor(img_shape)))),
            nn.Tanh()
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        img = self.model(z)
        img = img.view(img.size(0), *self.img_shape)
        return img


# 定义判别器
class Discriminator(nn.Module):
    def __init__(self, img_shape: Tuple[int, int, int]):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(int(torch.prod(torch.tensor(img_shape))), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)
        return validity


# 定义训练函数
def train(dataloader: DataLoader, discriminator: Discriminator, generator: Generator,
          device: torch.device, lr: float, b1: float, b2: float, latent_dim: int,
          n_epochs: int, sample_interval: int):
    # 损失函数
    adversarial_loss = nn.BCELoss()

    # 优化器
    optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(b1, b2))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(b1, b2))

    for epoch in range(n_epochs):
        for i, (imgs, _) in enumerate(dataloader):

            # 训练判别器
            real_imgs = imgs.to(device)
            z = torch.randn(imgs.size(0), latent_dim).to(device)
            fake_imgs = generator(z)

            real_loss = adversarial_loss(discriminator(real_imgs), torch.ones(real_imgs.size(0), 1).to(device))
            fake_loss = adversarial_loss(discriminator(fake_imgs.detach()),
                                         torch.zeros(fake_imgs.size(0), 1).to(device))
            d_loss = (real_loss + fake_loss) / 2

            optimizer_D.zero_grad()
            d_loss.backward()
            optimizer_D.step()

            # 训练生成器
            z = torch.randn(imgs.size(0), latent_dim).to(device)
            fake_imgs = generator(z)

            g_loss = adversarial_loss(discriminator(fake_imgs), torch.ones(fake_imgs.size(0), 1).to(device))

            optimizer_G.zero_grad()
            g_loss.backward()
            optimizer_G.step()

            print(
                f"[Epoch {epoch}/{n_epochs}] [Batch {i}/{len(dataloader)}] [D loss: {d_loss.item():.4f}] [G loss: {g_loss.item():.4f}]")

            batches_done = epoch * len(dataloader) + i
            if batches_done % sample_interval == 0:
                test(generator, device, latent_dim)


# 定义测试函数
def test(generator: Generator, device: torch.device, latent_dim: int):
    z = torch.randn(1, latent_dim).to(device)
    fake_img = generator(z)
    fake_img = fake_img.cpu().detach().numpy().squeeze()
    fake_img = (fake_img + 1) / 2 * 255
    fake_img = fake_img.astype('uint8')
    fake_img = fake_img.transpose(1, 2, 0)
    torchvision.utils.save_image(torch.from_numpy(fake_img), f"generated_{latent_dim}.png")


# 设置超参数
latent_dim = 100
lr = 0.0002
b1 = 0.5
b2 = 0.999
n_epochs = 200
batch_size = 64
sample_interval = 400
img_shape = (1, 28, 28)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载MNIST数据集
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 初始化生成器和判别器
generator = Generator(latent_dim, img_shape).to(device)
discriminator = Discriminator(img_shape).to(device)

# 训练GAN
train(dataloader, discriminator, generator, device, lr, b1, b2, latent_dim, n_epochs, sample_interval)
