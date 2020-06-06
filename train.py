import os
import sys
import pickle
import argparse
import numpy as np
import yaml

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
# from torch.optim.lr_scheduler import StepLR

from models.BaseModel import BaseModel
# from models.LogisticRegression import SKLRModel
from models.LR import LRModel
from models.LDA import LDAModel
from models.SVM import SVMModel
from models.LRRidge import LRRidgeModel
from models.LRLasso import LRLassoModel
from models.LeNet5 import LeNet5
from models.CNN import CNNModel


class dict2obj(dict):
    def __init__(self, d, default=None):
        self.__d = d
        self.__default = default
        super(self.__class__, self).__init__(d)

    def __getattr__(self, k):
        if k in self.__d:
            v = self.__d[k]
            if isinstance(v, dict):
                v = self.__class__(v)
            setattr(self, k, v)
            return v
        else:
            raise KeyError

def get_dataset(batch_size: int, test_batch_size: int):
    if not os.path.exists('./data/'):
        os.mkdir('./data/')


def get_dataset(batch_size: int, test_batch_size: int, train_set_size: int):
    if not os.path.exists('./data/'):
        os.mkdir('./data/')
    split_location = './data/mnist_train_split_%d.pl' % train_set_size
    if os.path.exists(split_location):
        with open(split_location, 'rb') as f:
            split_set: np.ndarray = pickle.load(f)
    else:
        import warnings
        warnings.warn('Creating MNIST Training Subset')
        import random
        mnist_trainset_index = list(range(60000))
        if 0 < train_set_size < 60000:
            split_set = np.array(random.choices(mnist_trainset_index, k=train_set_size))
            with open(split_location, 'wb') as f:
                pickle.dump(split_set, f)
        else:
            print('Using the whole training set')
            split_set = mnist_trainset_index

    train_dataset = datasets.MNIST('./data', train=True, download=False, transform=transforms.ToTensor())
    train_dataset.data = train_dataset.data[split_set]
    train_dataset.targets = train_dataset.targets[split_set]

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(
        datasets.MNIST('./data', train=False, transform=transforms.ToTensor()),
        batch_size=test_batch_size, shuffle=False)
    return train_loader, test_loader


if __name__ == "__main__":
    pass
    args = sys.argv
    assert args[1] in ('train', 'debug')
    if args[1] == 'train' or args[1] == 'debug':
        yaml_path = args[2]
        assert os.path.exists(yaml_path)
        with open(yaml_path, 'r') as f:
            configs = yaml.load(f, Loader=yaml.SafeLoader)
            print(configs)
        configs = dict2obj(configs)
        if args[1] == 'debug':
            configs.model.log_name = ''
    else:
        exit()

    use_cuda = configs['model']['use_cuda'] and torch.cuda.is_available()
    configs.device = torch.device("cuda" if use_cuda else "cpu")
    torch.manual_seed(configs.train.seed)

    train_loader, test_loader = get_dataset(configs.train.batch_size, configs.train.test_batch_size, configs.train.subset_size)

    model_abs :BaseModel  = globals()[configs.model.model_name]
    model = model_abs(configs)

    # model = LRModel('NNLogisticRegression', lr=1e-1, device=device).to(device)
    # model = SKLRModel('', lr=1e-1, device=device).to(device)
    # model = LDAModel('', lr=2e-1, device=device).to(device)
    # model = SVMModel('', lr=2e-1, device=device).to(device)
    # model = LRRidgeModel('NN+LR+Ridge', lr=1e-1, device=device).to(device)
    # model = LeNet5('LeNet5', lr=1e-3, device=device).to(device)

    # print(train_loader.dataset.data.shape)
    # print(test_loader.dataset.data.shape)

    model.run_epochs(epochs=configs.train.epochs, train_loader=train_loader, test_loader=test_loader)

    # if args.save_model:
    #     torch.save(model.state_dict(), "mnist_cnn.pt")

