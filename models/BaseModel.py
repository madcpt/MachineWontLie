import time
import torch
from torch import nn
from torch.nn import init
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from overrides import overrides


class BaseModel(nn.Module):
    def __init__(self, model_name: str, device: torch.device = torch.device('cpu')):
        super().__init__()
        # call 'super().__init__()' after the model has been constructed.
        self.local_time = time.localtime()
        if len(model_name) > 0:
            self.writer = SummaryWriter('runs/%s-%d.%d-%d:%d' % (
                model_name, self.local_time.tm_mon, self.local_time.tm_mday, self.local_time.tm_hour,
                self.local_time.tm_min))
        self.device = device
        # self.optimizer = torch.optim.SGD(self.parameters(), lr=lr)

    @overrides
    def forward(self, input_data: torch.Tensor) -> torch.Tensor:
        # data: batch x 1 x 28 x 28
        # implement forwarding
        # out: batch x 10
        pass

    @staticmethod
    def calc_loss(output_feature: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # out: batch x 10
        # target: batch
        pass

    def optimize(self, loss: torch.Tensor):
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.zero_grad()

    def train_epoch(self, epoch_num: int, train_loader: DataLoader):
        self.train()
        train_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            output = self.forward(data)
            loss = self.calc_loss(output, target)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            train_loss += loss.item()
            # if batch_idx % args.log_interval == 0:
            #     print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            #         epoch, batch_idx * len(data), len(train_loader.dataset),
            #                100. * batch_idx / len(train_loader), loss.item()))
        train_loss /= len(train_loader)
        self.writer.add_scalar('train_loss', train_loss, epoch_num)
        # self.writer.add_scalars('loss', {'train_loss': train_loss}, epoch_num)

    def test_epoch(self, epoch_num: int, test_loader: DataLoader):
        self.eval()
        test_loss, correct = 0, 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.forward(data)
                test_loss += self.calc_loss(output, target)
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)
        acc = 100. * correct / len(test_loader.dataset)
        print('Test at epoch {:d}: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
            epoch_num, test_loss, correct, len(test_loader.dataset), acc))
        self.writer.add_scalar('test_loss', test_loss, epoch_num)
        self.writer.add_scalar('test_acc', acc, epoch_num)
        # self.writer.add_scalars('loss', {'test_loss': test_loss}, epoch_num)

    def run_epochs(self, epochs: int, train_loader: DataLoader, test_loader: DataLoader):
        for epoch in range(epochs):
            self.train_epoch(epoch, train_loader)
            self.test_epoch(epoch, test_loader)

    @staticmethod
    def weight_init(m):
        """
        Usage:
            model = Model()
            model.apply(weight_init)
        """
        if isinstance(m, nn.Conv1d):
            init.normal_(m.weight.data)
            if m.bias is not None:
                init.normal_(m.bias.data)
        elif isinstance(m, nn.Conv2d):
            init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                init.normal_(m.bias.data)
        elif isinstance(m, nn.Conv3d):
            init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                init.normal_(m.bias.data)
        elif isinstance(m, nn.ConvTranspose1d):
            init.normal_(m.weight.data)
            if m.bias is not None:
                init.normal_(m.bias.data)
        elif isinstance(m, nn.ConvTranspose2d):
            init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                init.normal_(m.bias.data)
        elif isinstance(m, nn.ConvTranspose3d):
            init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                init.normal_(m.bias.data)
        elif isinstance(m, nn.BatchNorm1d):
            init.normal_(m.weight.data, mean=1, std=0.02)
            init.constant_(m.bias.data, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.normal_(m.weight.data, mean=1, std=0.02)
            init.constant_(m.bias.data, 0)
        elif isinstance(m, nn.BatchNorm3d):
            init.normal_(m.weight.data, mean=1, std=0.02)
            init.constant_(m.bias.data, 0)
        elif isinstance(m, nn.Linear):
            init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                init.normal_(m.bias.data)
        elif isinstance(m, nn.LSTM):
            for param in m.parameters():
                if len(param.shape) >= 2:
                    init.orthogonal_(param.data)
                else:
                    init.normal_(param.data)
        elif isinstance(m, nn.LSTMCell):
            for param in m.parameters():
                if len(param.shape) >= 2:
                    init.orthogonal_(param.data)
                else:
                    init.normal_(param.data)
        elif isinstance(m, nn.GRU):
            for param in m.parameters():
                if len(param.shape) >= 2:
                    init.orthogonal_(param.data)
                else:
                    init.normal_(param.data)
        elif isinstance(m, nn.GRUCell):
            for param in m.parameters():
                if len(param.shape) >= 2:
                    init.orthogonal_(param.data)
                else:
                    init.normal_(param.data)
