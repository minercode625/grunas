import numpy as np
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
import os
from torch.utils.data import Dataset
import cv2
from tqdm import tqdm
import albumentations
from albumentations.pytorch import ToTensorV2
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMG_SIZE = (256, 256) 
CROP_SIZE = (224, 224)

debug_mode = False

def get_loaders(train_portion, batch_size, path_to_save_data):
    img_per_class = 1300
    num_of_classes = 1000
    debug_size = 20
    label_file = open('train_labels.txt', mode="r")

    label_txt = label_file.readlines()

    label_file.close()

    file_list = os.listdir('ILSVRC2012_img_train')

    label_dict = {}
    for i in range(len(label_txt)):
        dir_str = label_txt[i].split()
        label_dict[dir_str[0]] = int(dir_str[1])
    x = []
    y = []


    for idx in tqdm(range(len(file_list))):
        item = file_list[idx]
        dir = '/home/cal-06/Desktop/test/ILSVRC2012_img_train/' + item
        img_list = os.listdir('/home/cal-06/Desktop/test/ILSVRC2012_img_train/' + item)
        for i, img_dir in enumerate(img_list):
            x.append(dir + '/' + img_dir)
            y.append(label_dict[item])
    
    y = torch.tensor(y).squeeze_()
    y -= 1
    train_transform = albumentations.Compose([
        albumentations.Resize(*IMG_SIZE),
        albumentations.RandomCrop(*CROP_SIZE),
        albumentations.HorizontalFlip(),
        albumentations.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ToTensorV2()
    ])
    train_data = ImagenetDataset(x, y, train_transform)

    num_train = len(train_data)
    indices = np.arange(num_train)
    np.random.shuffle(indices)
    split = int(np.floor(train_portion * num_train))

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
    debug_size = 100
    test_size = 50000
    label_file = open('validation_labels.txt', mode="r")

    label_txt = label_file.readlines()

    label_file.close()

    file_list = os.listdir('ILSVRC2012_img_val')
    x = []
    if debug_mode:
        y = np.zeros((debug_size, 1), dtype=np.int64)
    else:
        y = np.zeros((test_size, 1), dtype=np.int64)

    print("Test Data loading")

    for idx in tqdm(range(len(file_list))):
        if debug_mode and idx == debug_size:
            break
        item = file_list[idx]
        dir = 'ILSVRC2012_img_val/'+item
        item_idx = item.replace('ILSVRC2012_val_', '')
        item_idx = int(item_idx.replace('.JPEG', ''))
        x.append(dir)
        y[idx] = float(label_txt[item_idx - 1])
    y = y-1
    y = torch.tensor(y).squeeze_()
    test_transform = albumentations.Compose([
        albumentations.Resize(*IMG_SIZE),
        albumentations.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ToTensorV2()
    ])

    test_data = ImagenetDataset(x, y, test_transform)

    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size,
                                              shuffle=False, num_workers=16)
    return test_loader

class ImagenetDataset(Dataset):
    def __init__(self, x, y, transform):
        self.x = x
        self.y = y
        self.transform = transform
    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        image = cv2.imread(self.x[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return self.transform(image=image)['image'], self.y[idx]

