# -*- coding: utf-8 -*-
# Time    : 2023/10/29 15:26
# Author  : fanc
# File    : data_transform.py

import os
import numpy as np
import pandas as pd
import pydicom
import nrrd
import re
import json
import nibabel as nib
import SimpleITK as sitk
from tqdm import tqdm
def read_csv(file_path):
    df = pd.read_csv(file_path, encoding='gb18030')
    columns = list(df.columns)
    df.columns = ['姓名', '编号', '结石成分'] + columns[3:]
    df = df.dropna(subset=['结石成分']).reset_index(drop=True)
    return df[['姓名', '编号', '结石成分']]

def load_dcm(dcm_dir):
    dcm_list = os.listdir(dcm_dir)
    dicom_files = [pydicom.dcmread(os.path.join(dcm_list, f)) for f in sorted(os.listdir(dcm_list), reverse=True)]
    return dicom_files

def load_nrrd(file_path):
    data, header = nrrd.read(file_path)
    return data

def map_data(data_dir):
    infos = []
    # data_dir = r'C:\Users\Asus\Desktop\肺腺癌\data\肾结石数据\KdneyStone\202310326结石成分分析龙岗区人民医院李星智'
    dcm_dir = os.path.join(data_dir, '结石原始图像')
    nrrd_dir = os.path.join(data_dir, 'mask')
    csv_dir = os.path.join(data_dir, 'list.csv')
    # 读取csv文件
    df = read_csv(csv_dir)

    print(df)
    names = os.listdir(dcm_dir)
    for name in names:
        print(name)
        if re.search('\D+', name):
            temp = df[df['姓名'] == re.search('\D+', name).group()].reset_index(drop=True)
            #print('temp', temp)
            if len(temp) > 0:
                for i in range(len(temp)):
                    id_ = temp.loc[i, '编号']
                    if not os.path.exists(os.path.join(nrrd_dir, id_ + '.nrrd')):
                        continue
                    seg_data = load_nrrd(os.path.join(nrrd_dir, id_ + '.nrrd'))
                    seg_shape = seg_data.shape
                    temp_path = os.path.join(dcm_dir, name)
                    for j in os.listdir(temp_path):
                        if len(os.listdir(os.path.join(temp_path, j))) == seg_shape[-1]:
                            dic = {'name': re.search('\D+', name).group(), 'id': id_, 'shape': seg_shape,
                                   'label': int(temp.loc[i, '结石成分']),
                                   'dcm_dir': os.path.join(temp_path, j),
                                   'seg_dir': os.path.join(nrrd_dir, id_ + '.nrrd')}
                            infos.append(dic)
                            print(dic)
                            break
    infos.sort(key=lambda x: x['id'])

    print(len(infos))
    with open(os.path.join(data_dir, 'infos.json'), 'w') as f:
        json.dump(infos, f)

def makedirs(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
def data2nii(info_dir, save_dir):
    makedirs(os.path.join(save_dir, 'imgs_nii'))
    makedirs(os.path.join(save_dir, 'mask_nii'))
    makedirs(os.path.join(save_dir, 'slices_npy'))
    makedirs(os.path.join(save_dir, 'mask_npy'))
    with open(os.path.join(info_dir, 'infos.json'), 'r') as f:
        infos = json.load(f)
    # with open(os.path.join(info_dir, 'data', 'infos.json'), 'r') as f:
    #     old = json.load(f)
    i = 1
    files = os.listdir()
    for info in tqdm(infos):
        # if info in old:
        #     continue
        # else:
        seg_dir = info['seg_dir']
        filepath = os.path.join(info['dcm_dir'])
        series_id = sitk.ImageSeriesReader.GetGDCMSeriesIDs(filepath)
        series_file_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(filepath, series_id[0])
        series_reader = sitk.ImageSeriesReader()  # 读取数据端口
        series_reader.SetFileNames(series_file_names)  # 读取名称
        images = series_reader.Execute()
        sitk.WriteImage(images, os.path.join(save_dir, 'imgs_nii', f"{info['id']}.nii.gz"))  # 保存为nii
        # slices = [pydicom.dcmread(os.path.join(dcm_dir, i)) for i in os.listdir(dcm_dir)]
        # # Sort the dicom slices in their respective order
        # slices.sort(key=lambda x: int(x.InstanceNumber))
        # try:
        #     slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
        # except:
        #     slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)
        # for s in slices:
        #     s.SliceThickness = slice_thickness
        #
        # # Get the pixel values for all the slices
        # img = np.stack([s.pixel_array for s in slices])
        # img = img.astype(np.int16)
        # img[img == -2000] = 0
        # for slice_number in range(len(slices)):
        #     intercept = slices[slice_number].RescaleIntercept
        #     slope = slices[slice_number].RescaleSlope
        #     if slope != 1:
        #         img[slice_number] = slope * img[slice_number].astype(np.float64)
        #         img[slice_number] = img[slice_number].astype(np.int16)
        #     img[slice_number] += np.int16(intercept)
        # img = np.array(img, dtype=np.int16)
        # nifti_image = nib.Nifti1Image(img, affine=None)
        # nib.save(nifti_image, os.path.join(save_dir, 'imgs_nii', f"{info['id']}.nii.gz"))
        # np.save(os.path.join(save_dir, 'slices_npy', f"{info['id']}.npy"), slices)
        mask, header = nrrd.read(seg_dir)
        img = nib.Nifti1Image(mask, np.eye(4))
        nib.save(img, os.path.join(save_dir, 'mask_nii', f"{info['id']}-mask.nii.gz"))  # 保存 nii 文件
        # np.save(os.path.join(save_dir, 'mask_npy', f"{info['id']}-mask.npy"), mask)
        # print(f'完成进度：{i}/{len(infos)}--{i/len(infos)}')
        # i += 1

if __name__ == '__main__':
    # data_dir = r'C:\Users\Asus\Desktop\肺腺癌\data\肾结石数据\KdneyStone\202310326结石成分分析龙岗区人民医院李星智'
    data_dir = '/home/fanchenchenzc/datasets/202310326结石成分分析龙岗区人民医院李星智'
    # map_data(data_dir)
    data2nii(info_dir=data_dir, save_dir=data_dir)
