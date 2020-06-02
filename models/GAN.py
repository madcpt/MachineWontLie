import torch
from torch import nn
from torch.utils.data import DataLoader

from models.BaseModel import BaseModel
from overrides import overrides


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(784, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.main(x)


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(128, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 784),
            nn.Tanh()
        )

    def forward(self, x):
        return self.main(x)


class GAN(BaseModel):
    def __init__(self, model_name: str, lr=1e-1, device=torch.device('cpu')):
        super().__init__(model_name, device)
        self.generator = Generator()
        self.discriminator = Discriminator()
        self.optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=lr)
        self.optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=lr)

    @overrides
    def train_epoch(self, epoch_num: int, train_loader: DataLoader):
        self.train()
        train_loss = 0
        for batch_idx, (real_inputs, real_label) in enumerate(train_loader):
            real_inputs, real_label = real_inputs.to(self.device), real_label.to(self.device)
            real_outputs = self.discriminator.forward(real_inputs)

            noise = ((torch.rand(real_inputs.shape[0], 28 * 28) - 0.5) / 0.5).to(self.device)
            fake_inputs = self.generator(noise)
            fake_outputs = self.discriminator(fake_inputs)
            fake_label = torch.zeros(fake_inputs.shape[0], 1).to(self.device)

            outputs = torch.cat((real_outputs, fake_outputs), 0)
            targets = torch.cat((real_label, fake_label), 0)
            loss = self.calc_loss(outputs, targets)

            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        self.writer.add_scalar('train_loss', train_loss, epoch_num)
        # self.writer.add_scalars('loss', {'train_loss': train_loss}, epoch_num)
