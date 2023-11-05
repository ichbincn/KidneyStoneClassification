# -*- coding: utf-8 -*-
# Time    : 2023/10/30 16:03
# Author  : fanc
# File    : load_data.py

import os
import numpy as np
import random
import torch
from torch.utils.data import DataLoader, Dataset
import json
from skimage.transform import resize
from scipy.ndimage import zoom

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
    def __init__(self, data_dir, infos, resize_rate=0.25):
        slices_dir = os.path.join(data_dir, 'slices_npy')
        mask_dir = os.path.join(data_dir, 'mask_npy')

        self.resize_rate = resize_rate
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
        # img, mask = self.crop(img, mask)
        #img = resize(img, self.size2)
        #mask = resize(mask, self.size2)
        print(mask.shape, mask.sum())
        img = resize(img, (512, 512, 512))
        mask = resize(mask, (512, 512, 512), order=0)
        print(mask.shape, mask.sum())
        img = zoom(img, self.resize_rate, order=0)
        mask = zoom(mask, self.resize_rate, order=0, mode='nearest')
        # img = zoom(img, (1, 1, 0.25), order=0)
        # mask = zoom(mask, (1, 1, 0.25), order=0, mode='nearest')
        print(mask.shape, mask.sum())
        img = (img/255).astype(np.float32)
        if np.min(img) < np.max(img):
            img = img - np.min(img)
            img = img / np.max(img)
        img = torch.tensor(img).unsqueeze(0)
        mask = torch.tensor(mask, dtype=torch.uint8).unsqueeze(0)
        return img, mask

    def crop(self, image, mask):
        size = self.size
        # [w, h, d] = [(image.shape[i] - size[i]) // 2 for i in range(len(size))]
        mask_count = mask.sum()
        coordinates = np.argwhere(mask == 1)
        xx = max(coordinates.T[0]) - min(coordinates.T[0]) + 1
        yy = max(coordinates.T[1]) - min(coordinates.T[1]) + 1
        zz = max(coordinates.T[2]) - min(coordinates.T[2]) + 1
        if xx > size[0] or yy > size[1]:
            print("crop size is larger than image")
        # center coordinate
        x = int(xx / 2 + min(coordinates.T[0]))
        y = int(yy / 2 + min(coordinates.T[1]))
        # z = int(zz / 2 + min(coordinates.T[2]))

        w = x - size[0] // 2
        ww = w + size[0]
        h = y - size[1] // 2
        hh = h + size[1]
        # d = z - size[2] // 2
        # dd = d + size[2]

        if w < 0:
            w = 0
            ww = w + size[0]
        if h < 0:
            h = 0
            hh = h + size[1]
        # if z-size[2]//2 < 0:
        #     d = 0
        #     dd = d + size[2]

        if ww > mask.shape[0]:
            ww = -1
            w = ww - size[0]
        if hh > mask.shape[1]:
            hh = -1
            h = hh - size[1]
        # if z+size[2]//2 > mask.shape[2]:
        #     dd = -1
        #     d = size[2] - dd

        crop_mask = mask[w:ww, h:hh, :]
        # assert crop_mask.sum() == mask_count, "crop mask size is not equal to original mask"
        if crop_mask.sum() != mask_count:
            print("crop mask size is not equal to original mask")
        crop_img = image[w:ww, h:hh, :]
        return crop_img, crop_mask


def my_dataloader(data_dir, infos, batch_size=3, shuffle=True, num_workers=0, resize_rate=0.25):
    dataset = MyDataset(data_dir, infos, resize_rate=resize_rate)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return dataloader

# data_dir = r'C:\Users\Asus\Desktop\肺腺癌\data\肾结石数据\KdneyStone\202310326结石成分分析龙岗区人民医院李星智'
# #
# train_info, test_info = split_data(data_dir, rate=0.8)
# train_dataloader = my_dataloader(data_dir, train_info)
# test_dataloader = my_dataloader(data_dir, test_info)
# for i, (image, mask, label) in enumerate(train_dataloader):
#     pass
    # print(i,  image.shape, mask.shape, label)
    # print(mask.sum())
#
#
# for i, (image, mask, label) in enumerate(test_dataloader):
#     print(i,  image.shape, mask.shape, label)
