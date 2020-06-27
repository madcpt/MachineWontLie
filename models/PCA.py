import torch
from torch import nn
from torch.nn import init
from torch.utils.data import DataLoader
from overrides import overrides
import numpy as np
import time

from models.BaseModel import BaseModel


class PCAModel(BaseModel):
    def __init__(self, configs: object):
        super().__init__(configs.model.model_name, configs.device)
        from sklearn.decomposition import PCA
        self.pca_cls = PCA(n_components=30)

        from sklearn.svm import SVC
        self.svm_cls = SVC(kernel="rbf", probability=True, )

    @overrides
    def train_epoch(self, epoch_num: int, train_loader: DataLoader):
        x = torch.flatten(train_loader.dataset.data, 1).numpy()
        y = train_loader.dataset.targets.numpy()
        self.pca_cls.fit(x, y)
        x_pca = self.pca_cls.transform(x)
        # print(x_pca.shape)
        self.svm_cls.fit(x_pca, y)

    @overrides
    def test_epoch(self, epoch_num: int, test_loader: DataLoader):
        x = torch.flatten(test_loader.dataset.data, 1).numpy()
        y = test_loader.dataset.targets.numpy()
        pca_result: np.ndarray = self.pca_cls.transform(x)
        predict_score = self.svm_cls.predict(pca_result)
        predict_result = predict_score
        # predict_result = np.argmax(predict_score,axis=1)
        # print(x.shape, predict_score.shape, predict_result.shape, y.shape)
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
