import os
import sys
import pickle
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

from models.LogisticRegression import SKLRModel
from models.LR import LRModel
from models.LDA import LDAModel
from models.SVM import SVMModel


def get_dataset(batch_size: int, test_batch_size: int):
    if not os.path.exists('./data/'):
        os.mkdir('./data/')
    split_location = './data/mnist_train_split.pl'
    if os.path.exists(split_location):
        with open(split_location, 'rb') as f:
            split_set: np.ndarray = pickle.load(f)
    else:
        import warnings
        warnings.warn('Creating MNIST Training Subset')
        import random
        mnist_trainset_index = list(range(60000))
        split_set = np.array(random.choices(mnist_trainset_index, k=10000))
        with open(split_location, 'wb') as f:
            pickle.dump(split_set, f)

    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transforms.ToTensor())
    train_dataset.data = train_dataset.data[split_set]
    train_dataset.targets = train_dataset.targets[split_set]

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(
        datasets.MNIST('./data', train=False, transform=transforms.ToTensor()),
        batch_size=test_batch_size, shuffle=True)
    return train_loader, test_loader


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N', help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N', help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=30, metavar='N', help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR', help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M', help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N', help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False, help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    train_loader, test_loader = get_dataset(args.batch_size, args.test_batch_size)

    # model = LRModel('LogisticRegression', lr=1e-1, device=device).to(device)
    model = SKLRModel('', lr=1e-1, device=device).to(device)
    # model = LDAModel('', lr=2e-1, device=device).to(device)
    # model = SVMModel('', lr=2e-1, device=device).to(device)

    # print(train_loader.dataset.data.shape)
    # print(test_loader.dataset.data.shape)

    model.run_epochs(epochs=args.epochs, train_loader=train_loader, test_loader=test_loader)

    # if args.save_model:
    #     torch.save(model.state_dict(), "mnist_cnn.pt")


if __name__ == '__main__':
    main()
