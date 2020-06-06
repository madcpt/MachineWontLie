import torch
from torch import nn
from torch.nn import functional as F
from overrides import overrides

from models.BaseModel import BaseModel


class LeNet5(BaseModel):
    def __init__(self, configs: object):
        super().__init__(configs.model.model_name, configs.device)
        self.conv1 = nn.Conv2d(1, 6, 5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self = self.to(device=configs.device)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=configs.train.learning_rate)
        self.apply(self.weight_init)
    
    @overrides
    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    #     self.conv1 = nn.Conv2d(1, 6, 5, padding=2)
    #     self.relu1 = nn.ReLU()
    #     self.pool1 = nn.MaxPool2d(2)
    #     self.conv2 = nn.Conv2d(6, 16, 5)
    #     self.relu2 = nn.ReLU()
    #     self.pool2 = nn.MaxPool2d(2)
    #     self.fc1 = nn.Linear(256, 120)
    #     self.relu3 = nn.ReLU()
    #     self.fc2 = nn.Linear(120, 84)
    #     self.relu4 = nn.ReLU()
    #     self.fc3 = nn.Linear(84, 10)
    #     self.relu5 = nn.ReLU()
    #     self = self.to(device=configs.device)
    #     self.optimizer = torch.optim.SGD(self.parameters(), lr=configs.train.learning_rate)
    #     self.apply(self.weight_init)

    # @overrides
    # def forward(self, input_data: torch.Tensor) -> torch.Tensor:
    #     y = self.conv1(input_data.float())
    #     y = self.relu1(y)
    #     y = self.pool1(y)
    #     y = self.conv2(y)
    #     y = self.relu2(y)
    #     y = self.pool2(y)
    #     y = y.view(y.shape[0], -1)
    #     y = self.fc1(y)
    #     y = self.relu3(y)
    #     y = self.fc2(y)
    #     y = self.relu4(y)
    #     y = self.fc3(y)
    #     y = self.relu5(y)
    #     return y

    @staticmethod
    def calc_loss(output_feature: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # loss = torch.nn.functional.nll_loss(torch.log_softmax(output_feature, dim=-1), target)
        # return loss
        loss = nn.CrossEntropyLoss()(output_feature, target.long())
        return loss

        batch_size, _ = output_feature.shape
        output_feature = torch.softmax(output_feature, dim=-1)
        target_flatten = target.unsqueeze(dim=-1)
        log_likelihood = torch.gather(output_feature, 1, target_flatten)
        loss = -torch.sum(log_likelihood)

        # target_flatten = nn.functional.one_hot(target, num_classes=10)
        # output_feature = torch.softmax(output_feature, dim=-1)
        # loss = torch.nn.functional.mse_loss(output_feature, target_flatten.float())
        return loss

