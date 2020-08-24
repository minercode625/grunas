import numpy as np
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
import os

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_loaders(train_portion, batch_size, path_to_save_data):
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    train_data = datasets.ImageNet(root=path_to_save_data, train=True,
                                  download=True, transform=train_transform)

    num_train = len(train_data)  # 50k
    indices = list(range(num_train))  #
    split = int(np.floor(train_portion * num_train))  # 40k

    train_idx, valid_idx = indices[:split], indices[split:]

    train_sampler = SubsetRandomSampler(train_idx)

    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size, sampler=train_sampler, num_workers=2)

    if train_portion == 1:
        return train_loader

    valid_sampler = SubsetRandomSampler(valid_idx)

    val_loader = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size, sampler=valid_sampler, num_workers=2)

    return train_loader, val_loader


def get_test_loader(batch_size, path_to_save_data):
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    test_data = datasets.ImageNet(root=path_to_save_data, train=False,
                                 download=True, transform=test_transform)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size,
                                              shuffle=False, num_workers=16)
    return test_loader
