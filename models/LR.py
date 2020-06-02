import torch
from torch import nn
from overrides import overrides

from models.BaseModel import BaseModel


class LRModel(BaseModel):
    def __init__(self, model_name: str, lr: float, device: torch.device):
        super().__init__(model_name, device)
        self.fc = nn.Linear(28 * 28, 10, bias=True).to(self.device)
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

