import torch
from torch import nn
from torch.nn import init
from torch.utils.data import DataLoader
from overrides import overrides
import numpy as np

from models.BaseModel import BaseModel


class LDAModel(BaseModel):
    def __init__(self, model_name: str, lr: float, device: torch.device):
        super().__init__(model_name, torch.device('cpu'))
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        self.lda_cls = LinearDiscriminantAnalysis()

    @overrides
    def train_epoch(self, epoch_num: int, train_loader: DataLoader):
        x = torch.flatten(train_loader.dataset.data, 1).numpy()
        y = train_loader.dataset.targets.numpy()
        self.lda_cls.fit(x, y)

    @overrides
    def test_epoch(self, epoch_num: int, test_loader: DataLoader):
        x = torch.flatten(test_loader.dataset.data, 1).numpy()
        y = test_loader.dataset.targets.numpy()
        predict_result: np.ndarray = self.lda_cls.predict(x)
        results: np.ndarray = predict_result == y
        return sum(results) / len(results)

    @overrides
    def run_epochs(self, epochs: int, train_loader: DataLoader, test_loader: DataLoader):
        self.train_epoch(0, train_loader)
        acc = self.test_epoch(0, test_loader)
        print(acc)
