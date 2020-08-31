import numpy as np
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
import os
from torch.utils.data import Dataset
import cv2
from tqdm import tqdm

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMG_SIZE = int(256 / 4)
CROP_SIZE = int(224 / 4)

def get_loaders(train_portion, batch_size, path_to_save_data):

    label_file = open('train_labels.txt', mode="r")

    label_txt = label_file.readlines()

    label_file.close()

    file_list = os.listdir('ILSVRC2012_img_train')

    label_dict = {}
    for i in range(len(label_txt)):
        dir_str = label_txt[i].split()
        label_dict[dir_str[0]] = int(dir_str[1])

    x = np.zeros((1300 * 1000, IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)
    y = np.zeros((1300 *1000, 1), dtype=np.float32)

    print("Training Data loading")
    for idx in tqdm(range(len(file_list))):
        item = file_list[idx]
        dir = 'ILSVRC2012_img_train/'+item
        img_list = os.listdir('ILSVRC2012_img_train/'+item)
        for i, img_dir in enumerate(img_list):
            img = cv2.imread(dir + '/' + img_dir)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            save_idx = 1300 * idx + i
            x[save_idx, :, : , :] = img
            y[save_idx, :] = label_dict[item]

    train_transform = transforms.Compose([
        transforms.CenterCrop(CROP_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
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

    label_file = open('validation_labels.txt', mode="r")

    label_txt = label_file.readlines()

    label_file.close()

    file_list = os.listdir('ILSVRC2012_img_val')

    x = np.zeros((50000, IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)
    y = np.zeros((50000, 1), dtype=np.float32)

    print("Test Data loading")

    for idx in tqdm(range(len(file_list))):
        item = file_list[idx]
        dir = 'ILSVRC2012_img_val/'+item
        item_idx = item.replace('ILSVRC2012_val_', '')
        item_idx = int(item_idx.replace('.JPEG', ''))
        img = cv2.imread(dir)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        x[idx, :, :, :] = img
        y[idx, :] = float(label_txt[item_idx - 1])
        

    test_transform = transforms.Compose([
        transforms.CenterCrop(CROP_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
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
        self.target_transform = transforms.ToTensor()

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return self.transform(self.x[idx]), self.target_transform(self.y[idx])

