# -*- coding: utf-8 -*-
# Time    : 2023/10/30 16:03
# Author  : fanc
# File    : load_data.py

import os
import re

import numpy as np
import random
import torch
from torch.utils.data import DataLoader, Dataset
import json
from skimage.transform import resize
from scipy.ndimage import zoom
import SimpleITK as sitk
from scipy.ndimage import zoom

from collections import defaultdict


def split_data(data_dir, rate=0.8):
    with open(os.path.join(data_dir, 'infos.json'), 'r') as f:
        infos = json.load(f)

    # 创建一个字典，用于按类别存储数据
    class_data = defaultdict(list)
    for info in infos:
        label = info['label']  # 假设数据集中每个样本都有'label'字段表示类别
        class_data[label].append(info)

    train_infos = []
    test_infos = []

    # 对每个类别进行分层抽样
    for label, data in class_data.items():
        random.seed(1900)
        random.shuffle(data)
        num_samples = len(data)
        train_num = int(rate * num_samples)
        train_infos.extend(data[:train_num])
        test_infos.extend(data[train_num:])

    return train_infos, test_infos

class MyDataset(Dataset):
    def __init__(self, data_dir, infos, input_size, phase='train', task=[0, 1]):
        '''
        task: 0 :cla,  1 :seg
        '''
        img_dir = os.path.join(data_dir, 'imgs_nii')
        mask_dir = os.path.join(data_dir, 'mask_nii')

        self.input_size = tuple([int(i) for i in re.findall('\d+', str(input_size))])
        self.img_dir = img_dir
        if 0 in task:
            self.labels = [i['label'] for i in infos]
        if 1 in task:
            self.mask_dir = mask_dir
        # self.labels = [[1, 0] if int(i['label']) == 1 else [0, 1] for i in infos]

        self.ids = [i['id'] for i in infos]
        self.phase = phase

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        img = sitk.ReadImage(os.path.join(self.img_dir, f"{self.ids[i]}.nii.gz"))
        mask = sitk.ReadImage(os.path.join(self.mask_dir, f"{self.ids[i]}-mask.nii.gz"))
        if self.phase == 'train':
            img, mask = self.train_preprocess(img, mask)
        else:
            img, mask = self.val_preprocess(img, mask)
        label = self.labels[i]

        img = torch.tensor(img, dtype=torch.float32).unsqueeze(0)
        mask = torch.tensor(mask, dtype=torch.uint8).unsqueeze(0)

        return img, mask, label

    def train_preprocess(self, img, mask):
        img = self.resample(img)
        mask = self.resample(mask)
        # print(img.shape, mask.shape)
        assert img.shape == mask.shape, "img and mask shape not match"
        img, mask = self.crop(img, mask)
        img = self.normalize(img)
        img, mask = self.resize(img, mask)

        return img, mask
    def val_preprocess(self, img, mask):
        img = self.resample(img)
        mask = self.resample(mask)
        assert img.shape == mask.shape, "img and mask shape not match"
        # img, mask = self.crop(img, mask)
        img = self.normalize(img)
        img, mask = self.resize(img, mask)

        return img, mask

    def crop(self, img, mask):
        crop_img = img
        crop_mask = mask
        # amos kidney mask
        crop_mask[crop_mask == 2] = 1
        crop_mask[crop_mask != 1] = 0
        target = np.where(crop_mask == 1)
        [d, h, w] = crop_img.shape
        [max_d, max_h, max_w] = np.max(np.array(target), axis=1)
        [min_d, min_h, min_w] = np.min(np.array(target), axis=1)
        [target_d, target_h, target_w] = np.array([max_d, max_h, max_w]) - np.array([min_d, min_h, min_w])
        z_min = int((min_d - target_d / 2) * random.random())
        y_min = int((min_h - target_h / 2) * random.random())
        x_min = int((min_w - target_w / 2) * random.random())

        z_max = int(d - ((d - (max_d + target_d / 2)) * random.random()))
        y_max = int(h - ((h - (max_h + target_h / 2)) * random.random()))
        x_max = int(w - ((w - (max_w + target_w / 2)) * random.random()))

        z_min = np.max([0, z_min])
        y_min = np.max([0, y_min])
        x_min = np.max([0, x_min])

        z_max = np.min([d, z_max])
        y_max = np.min([h, y_max])
        x_max = np.min([w, x_max])

        z_min = int(z_min)
        y_min = int(y_min)
        x_min = int(x_min)

        z_max = int(z_max)
        y_max = int(y_max)
        x_max = int(x_max)
        crop_img = crop_img[z_min: z_max, y_min: y_max, x_min: x_max]
        crop_mask = crop_mask[z_min: z_max, y_min: y_max, x_min: x_max]

        return crop_img, crop_mask

    def resample(self, itkimage, new_spacing=[1, 1, 1]):
        # spacing = itkimage.GetSpacing()
        img_array = sitk.GetArrayFromImage(itkimage)
        # resize_factor = spacing / np.array(new_spacing)
        # new_real_shape = img_array.shape * resize_factor
        # new_shape = np.round(new_real_shape)
        # real_resize_factor = new_shape / img_array.shape
        # new_spacing = spacing / real_resize_factor
        # img = zoom(img_array, real_resize_factor, mode='nearest')
        # resampler = sitk.ResampleImageFilter()
        # originSize = itkimage.GetSize()  # 原来的体素块尺寸
        # # newSize = np.array(newSize, float)
        # factor = spacing / new_spacing
        # resampler.SetReferenceImage(itkimage)  # 需要重新采样的目标图像
        # resampler.SetOutputSpacing(new_spacing)
        # resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
        # resampler.SetInterpolator(resamplemethod)
        # itkimgResampled = resampler.Execute(itkimage)  # 得到重新采样后的图像
        return np.array(img_array, dtype=np.float32)

    def normalize(self, img):
        std = np.std(img)
        avg = np.average(img)
        return (img - avg + std) / (std * 2)

    def resize(self, img, mask):
        img = np.transpose(img, (2, 1, 0))
        mask = np.transpose(mask, (2, 1, 0))
        rate = np.array(self.input_size) / np.array(img.shape)
        img = zoom(img, rate.tolist(), order=0)
        mask = zoom(mask, rate.tolist(), order=0, mode='nearest')
        return img, mask



def my_dataloader(data_dir, infos, batch_size=3, shuffle=True, num_workers=0, input_size=(128, 128, 128)):
    dataset = MyDataset(data_dir, infos, input_size=input_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return dataloader

# data_dir = r'C:\Users\Asus\Desktop\data'
# #
# train_info, test_info = split_data(data_dir, rate=0.8)
# print(len(train_info), len(test_info))
# train_dataloader = my_dataloader(data_dir, train_info, input_size=(128, 128, 128))
# test_dataloader = my_dataloader(data_dir, test_info, input_size=(128, 128, 128))
# for i, (image, mask, label) in enumerate(train_dataloader):
#     print(i,image.shape, mask.shape, label)
#     break
#     print(mask.sum())
# #
#
# for i, (image, mask, label) in enumerate(test_dataloader):
#     print(i,  image.shape, mask.shape, label)
