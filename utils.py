import os
import matplotlib
from sklearn.model_selection import train_test_split
from MyDataset import MyDataset
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as transforms
# from torchvision.datasets.utils import download_url
from torchvision.datasets import CIFAR100
from torch.utils.data import DataLoader, random_split

# pre-computed in another file "cifar_mean_std.py"
training_set_mean = (0.5071, 0.4865, 0.4409)
training_set_std = (0.2673, 0.2564, 0.2762)
def get_train_valid_loader(dataset_dir, batch_size, mixup, seed, save_images):
    training_transformation = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(training_set_mean, training_set_std)
    ])
    valid_transformation = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(training_set_mean, training_set_std)
    ])
    train_set = CIFAR100(root=dataset_dir, train=True, download=True) # transform=

    train_set, validation_set = random_split(train_set, [40000, 10000], generator=torch.Generator().manual_seed(seed))
    train_set = MyDataset(train_set, transform=training_transformation)
    validation_set = MyDataset(validation_set, transform=valid_transformation)
    train_loader = DataLoader(train_set, batch_size=batch_size)
    valid_loader = DataLoader(validation_set, batch_size=batch_size)
    return train_loader, valid_loader

def get_test_loader(dataset_dir, batch_size):
    transformation = transforms.Compose([
        # transforms.RandomCrop(32, padding=4),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(training_set_mean, training_set_std)
    ])
    test_set = CIFAR100(root=dataset_dir, train=False, download=True, transform=transformation)
    test_loader = DataLoader(test_set, batch_size=batch_size)
    return test_loader


def onehot(label, n_classes):
    return torch.zeros(label.size(0), n_classes).scatter_(
        1, label.view(-1, 1), 1)


def mixup(X, y, alpha=0.2, n_classes=100):
    idx = torch.randperm(X.size(0))
    X2 = X[idx]
    y2 = y[idx]

    y = onehot(y, n_classes)
    y2 = onehot(y2, n_classes)

    lam = torch.FloatTensor([np.random.beta(alpha, alpha)])
    mixed_X = X * lam + X2 * (1 - lam)
    mixed_y = y * lam + y2 * (1 - lam)

    return mixed_X, mixed_y

def plot_loss_acc(train_loss, val_loss, train_acc, val_acc, fig_name):
    x = np.arange(len(train_loss))
    max_loss = max(max(train_loss), max(val_loss))

    fig, ax1 = plt.subplots()
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('loss')
    ax1.set_ylim([0,max_loss+1])
    yticks = np.linspace(0, max_loss + 1, 20)
    ax1.set_yticks(yticks)
    lns1 = ax1.plot(x, train_loss, 'yo-', label='train_loss')
    lns2 = ax1.plot(x, val_loss, 'go-', label='val_loss')
    # ax1.tick_params(axis='y', labelcolor='tab:blue')

    ax2 = ax1.twinx()
    ax2.set_ylabel('accuracy')
    ax2.set_ylim([0,1])
    yticks = np.linspace(0, 1, 30)
    ax2.set_yticks(yticks)
    lns3 = ax2.plot(x, train_acc, 'bo-', label='train_acc')
    lns4 = ax2.plot(x, val_acc, 'ro-', label='val_acc')
    # ax2.tick_params(axis='y', labelcolor='tab:red')

    lns = lns1+lns2+lns3+lns4
    labs = [l.get_label() for l in lns]
    ax2.legend(lns, labs, loc=0)

    fig.tight_layout()
    plt.title(fig_name)

    plt.savefig(os.path.join('./diagram', fig_name))

    np.savez(os.path.join('./diagram', fig_name.replace('.png ', '.npz')), train_loss=train_loss, val_loss=val_loss, train_acc=train_acc, val_acc=val_acc)
