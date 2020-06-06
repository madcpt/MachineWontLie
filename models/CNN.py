import torch
from torch import nn
from overrides import overrides

from models.BaseModel import BaseModel


class CNNModel(BaseModel):
    def __init__(self, configs: object):
        super().__init__(configs.model.model_name, configs.device)
        self.conv1 = nn.Sequential(         # input shape (1, 28, 28)
            nn.Conv2d(
                in_channels=1,              # input height
                out_channels=16,            # n_filters
                kernel_size=5,              # filter size
                stride=1,                   # filter movement/step
                padding=2,                  # if want same width and length of this image after Conv2d, padding=(kernel_size-1)/2 if stride=1
            ),                              # output shape (16, 28, 28)
            nn.ReLU(),                      # activation
            nn.MaxPool2d(kernel_size=2),    # choose max value in 2x2 area, output shape (16, 14, 14)
        )
        self.conv2 = nn.Sequential(         # input shape (16, 14, 14)
            nn.Conv2d(16, 32, 5, 1, 2),     # output shape (32, 14, 14)
            nn.ReLU(),                      # activation
            nn.MaxPool2d(2),                # output shape (32, 7, 7)
        )
        self.out = nn.Linear(32 * 7 * 7, 10)   # fully connected layer, output 10 classes
        self = self.to(device=configs.device)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=configs.train.learning_rate)
        self.apply(self.weight_init)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)           # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        output = self.out(x)
        return output    # return x for visualization

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

