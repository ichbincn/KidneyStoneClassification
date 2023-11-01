# -*- coding: utf-8 -*-
# Time    : 2023/10/30 16:03
# Author  : fanc
# File    : load_data.py

import os
import numpy as np
import random
from torch.utils.data import DataLoader, Dataset
import json
from skimage.transform import resize

def split_data(data_dir, rate=0.8):
    with open(os.path.join(data_dir, 'infos.json'), 'r') as f:
        infos = json.load(f)
    random.seed(1900)
    random.shuffle(infos)
    num_samples = len(infos)
    train_num = int(rate * num_samples)
    test_num = num_samples - train_num
    train_infos = infos[:train_num]
    test_infos = infos[train_num:]

    return train_infos, test_infos

class MyDataset(Dataset):
    def __init__(self, data_dir, infos, size=(256, 256, 256)):
        slices_dir = os.path.join(data_dir, 'slices_npy')
        mask_dir = os.path.join(data_dir, 'mask_npy')

        self.size = size
        self.slices_dir = slices_dir
        self.mask_dir = mask_dir
        self.labels = [i['label'] for i in infos]
        self.ids = [i['id'] for i in infos]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        image = np.load(os.path.join(self.slices_dir, f"{self.ids[i]}.npy")).astype('float64')
        mask = np.load(os.path.join(self.mask_dir, f"{self.ids[i]}-mask.npy")).astype('float64')
        image, mask = self.preprocess(image, mask)
        label = self.labels[i]
        return image, mask, label
    def preprocess(self, img, mask):
        img = resize(img, self.size)
        mask = resize(mask, self.size)
        img = (img/255).astype(np.float32)
        if np.min(img) < np.max(img):
            img = img - np.min(img)
            img = img / np.max(img)
        return img, mask

def my_dataloader(data_dir, infos, batch_size=3, shuffle=True, num_workers=0, size=(256, 256, 256)):
    dataset = MyDataset(data_dir, infos, size=size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return dataloader

# data_dir = r'C:\Users\Asus\Desktop\肺腺癌\data\肾结石数据\KdneyStone\202310326结石成分分析龙岗区人民医院李星智'
#
# train_info, test_info = split_data(data_dir, rate=0.8)
# train_dataloader = my_dataloader(data_dir, train_info)
# test_dataloader = my_dataloader(data_dir, test_info)
# for i, (image, mask, label) in enumerate(train_dataloader):
#     print(i,  image.shape, mask.shape, label)
#
# for i, (image, mask, label) in enumerate(test_dataloader):
#     print(i,  image.shape, mask.shape, label)
