import os
import numpy as np
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR100
from torch.utils.data import DataLoader, random_split
import torch

def compute_mean_std(loader, train_set):
    """compute the mean and std of cifar100 dataset
    Args:
        cifar100_training_dataset or cifar100_test_dataset
        witch derived from class torch.utils.data

    Returns:
        a tuple contains mean, std value of entire dataset
    """

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    h, w = 0, 0
    for batch_idx, (inputs, targets) in enumerate(loader):
        inputs = inputs.to(device)
        if batch_idx == 0:
            h, w = inputs.size(2), inputs.size(3)
            print(inputs.min(), inputs.max())
            chsum = inputs.sum(dim=(0, 2, 3), keepdim=True)
        else:
            chsum += inputs.sum(dim=(0, 2, 3), keepdim=True)
    mean = chsum / len(train_set) / h / w
    print('mean: %s' % mean.view(-1))

    chsum = None
    for batch_idx, (inputs, targets) in enumerate(loader):
        inputs = inputs.to(device)
        if batch_idx == 0:
            chsum = (inputs - mean).pow(2).sum(dim=(0, 2, 3), keepdim=True)
        else:
            chsum += (inputs - mean).pow(2).sum(dim=(0, 2, 3), keepdim=True)
    std = torch.sqrt(chsum / (len(train_set) * h * w - 1))
    print('std: %s' % std.view(-1))

    print('Done!')

def get_train_valid_loader(dataset_dir, batch_size):
    # download dataset from amazaon
    # if not os.path.exists(dataset_dir):
    #     os.makedirs(dataset_dir)
    train_set = CIFAR100(root=dataset_dir, train=True, download=True, transform=transforms.ToTensor())
    # train_set, validation_set = random_split(train_set, [40000, 10000], generator=torch.Generator().manual_seed(0))
    train_loader = DataLoader(train_set, batch_size=batch_size)
    return train_loader, train_set

if __name__ == '__main__':
    train_loader, train_set = get_train_valid_loader("./dataset", batch_size=10)
    compute_mean_std(train_loader, train_set)
