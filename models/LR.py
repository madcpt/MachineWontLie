import torch
from torch import nn
from torch.nn import init
from overrides import overrides

from models.BaseModel import BaseModel


class LRModel(BaseModel):
    def __init__(self, model_name: str, lr: float, device: torch.device):
        super().__init__(model_name, device)
        self.fc = nn.Linear(28 * 28, 10).to(self.device)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=lr)
        self.apply(self.weight_init)

    @overrides
    def forward(self, input_data: torch.Tensor) -> torch.Tensor:
        # self.writer.add_image('raw_pic', input_data[0:2], 0)
        # self.writer.add_image('raw_pic_1', input_data[1], 1)
        # for i in range(input_data.size(0)):
        #     self.writer.add_image('raw_pic', input_data[i], i)
        x = torch.flatten(input_data, 1)
        x = self.fc(x)
        # x = nn.functional.sigmoid(x)
        return x

    @staticmethod
    def calc_loss(output_feature: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # return torch.nn.functional.nll_loss(output_feature, target)
        batch_size, _ = output_feature.shape
        target_flatten = nn.functional.one_hot(target, num_classes=10)
        output_feature = torch.sigmoid(output_feature)
        loss = torch.nn.functional.binary_cross_entropy(output_feature.view(-1, 1), target_flatten.view(-1, 1).float())
        return loss

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
