import torch
from torch import nn
from torch.utils.data import DataLoader
import numpy as np

from models.BaseModel import BaseModel
from overrides import overrides
import matplotlib.pyplot as plt

plt.rcParams['image.cmap'] = 'gray'

def show_images(images):
    sqrtn = int(np.ceil(np.sqrt(images.shape[0])))

    for index, image in enumerate(images):
        plt.subplot(sqrtn, sqrtn, index+1)
        plt.imshow(image.reshape(28, 28))

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
    def __init__(self, configs: object):
        super().__init__(configs.model.model_name, configs.device)
        self.generator = Generator()
        self.discriminator = Discriminator()
        self.optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=configs.train.generator_learning_rate, betas=(0.5, 0.999))
        self.optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=configs.train.discriminator_learning_rate, betas=(0.5, 0.999))
        self.criterion = nn.BCELoss()

    @overrides
    def train_epoch(self, epoch_num: int, train_loader: DataLoader):
        self.train()
        D_loss, G_loss = 0, 0
        for batch_idx, (real_inputs, real_label) in enumerate(train_loader):
            real_inputs = real_inputs.view(-1, 784).to(self.device)
            noise = ((torch.rand(real_inputs.shape[0], 128) - 0.5) / 0.5).to(self.device)

            # Discriminator
            real_outputs = self.discriminator(real_inputs)
            fake_inputs = self.generator(noise)
            fake_outputs = self.discriminator(fake_inputs)
            real_label_D = torch.ones(fake_inputs.shape[0], 1).to(self.device)
            fake_label_D = torch.zeros(fake_inputs.shape[0], 1).to(self.device)

            outputs_D = torch.cat((real_outputs, fake_outputs), 0)
            targets_D = torch.cat((real_label_D, fake_label_D), 0)
            self.optimizer_D.zero_grad()
            loss_D = self.calc_loss(outputs_D, targets_D)
            loss_D.backward()
            self.optimizer_D.step()

            # Generator
            noise = ((torch.rand(real_inputs.shape[0], 128) - 0.5) / 0.5).to(self.device)
            fake_inputs = self.generator(noise)
            fake_outputs = self.discriminator(fake_inputs)
            fake_label_G = torch.ones(fake_inputs.shape[0], 1).to(self.device)
            loss_G = self.calc_loss(fake_outputs, fake_label_G)
            self.optimizer_G.zero_grad()
            loss_G.backward()
            self.optimizer_G.step()

            D_loss += loss_D.item()
            G_loss += loss_G.item()

        imgs_numpy = (fake_inputs.data.cpu().numpy()+1.0)/2.0
        show_images(imgs_numpy[0:9])

        plt.savefig('./runs/GAN-pic/epoch-%d' % epoch_num)
        # if batch_idx == 0 and (epoch_num >= 5 and epoch_num % 5 == 0):
        #     plt.show()
        D_loss /= len(train_loader)
        G_loss /= len(train_loader)
        # if self.writer:
        #     self.writer.add_scalar('train_loss', train_loss, epoch_num)
        print('D_loss: %.4f, G_loss: %.4f' %(D_loss, G_loss))
        if self.writer:
            self.writer.add_scalars('loss', {'D_loss': D_loss, 'G_loss': G_loss}, epoch_num)
    
    @overrides
    def calc_loss(self, output_feature: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.criterion(output_feature, target)

    @overrides
    def run_epochs(self, epochs: int, train_loader: DataLoader, test_loader: DataLoader):
        for epoch in range(epochs):
            self.train_epoch(epoch, train_loader)
            # self.test_epoch(epoch, test_loader)