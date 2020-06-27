import torch
from torch import nn
from torch.nn import init
from torch.utils.data import DataLoader
from overrides import overrides
import numpy as np
import time

from models.BaseModel import BaseModel


class LDAModel(BaseModel):
    def __init__(self, configs: object):
        super().__init__(configs.model.model_name, configs.device)
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
        t1 = time.time()
        self.train_epoch(0, train_loader)
        t2 = time.time()
        acc = self.test_epoch(0, test_loader)
        if self.writer:
            self.writer.add_scalar('test_acc', acc, 0)
        print(acc, t2 - t1, time.time() - t2)
