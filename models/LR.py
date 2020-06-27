import torch
from torch import nn
from overrides import overrides

from models.BaseModel import BaseModel


class LRModel(BaseModel):
    def __init__(self, configs: object):
        super().__init__(configs.model.model_name, configs.device)
        self.fc = nn.Linear(28 * 28, 10, bias=True).to(configs.device)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=configs.train.learning_rate)
        self.apply(self.weight_init)
        self.criterion = nn.CrossEntropyLoss()

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

    @overrides
    def calc_loss(self, output_feature: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # return torch.nn.functional.nll_loss(output_feature, target)
        batch_size, _ = output_feature.shape
        # FIXME
        # target_flatten = nn.functional.one_hot(target, num_classes=10)
        # output_feature = torch.sigmoid(output_feature)
        # loss = torch.nn.functional.binary_cross_entropy(output_feature.view(-1, 1), target_flatten.view(-1, 1).float())
        # TODO: correctness unchecked
        # output_feature = torch.softmax(output_feature, dim=-1)
        # target_flatten = target.unsqueeze(dim=-1)
        # log_likelihood = torch.gather(output_feature, 1, target_flatten)
        # loss = -torch.sum(log_likelihood)
        loss = self.criterion(output_feature, target)
        return loss

