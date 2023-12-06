"""
@File  :segment_data.py
@Author:chenyihan
@Date  :2023/12/5 16:28
@Desc  用于图像语义分割的数据集接口:
"""
import os.path

import torch
import os.path as op
import numpy as np
import torch.utils.data as data
import cv2
from torchvision import transforms
from sklearn.model_selection import train_test_split
from albumentations import Compose, Resize, Normalize, ColorJitter
from albumentations.pytorch import ToTensorV2
import json


class SegmentData(data.Dataset):
    def __init__(self, data_dir=r'../dataset/scratch',
                 class_num=1,
                 train=True,
                 no_augment=True,
                 aug_prob=0.5):
        # Set all input args as attributes
        self.__dict__.update(locals())
        self.aug = train and not no_augment
        self.check_files()

    def check_files(self):
        with open(os.path.join(self.data_dir,'data.json'), 'r', encoding='UTF-8') as f:
            config = json.load(f)
        train_txt_path = config["train_path"]
        valid_txt_path = config["valid_path"]
        with open(train_txt_path, 'r', encoding='UTF-8') as train_f:
            img_list_train = [line.strip("\n").split(" ") for line in train_f.readlines()]
        with open(valid_txt_path, 'r', encoding='UTF-8') as val_f:
            img_list_val = [line.strip("\n").split(" ") for line in val_f.readlines()]
        self.path_list = img_list_train if self.train else img_list_val
        self.crop_size = config["crop_size"]
        self.img_mean = config["img_mean"]
        self.img_std = config["img_std"]
        self.is_label_divide = config["is_label_divide"]

    def __len__(self):
        return len(self.path_list)

    def __getitem__(self, idx):
        img_path, label_path = self.path_list[idx]
        img_path, label_path = os.path.join(self.data_dir, img_path), os.path.join(self.data_dir, label_path)
        filename = op.splitext(op.basename(img_path))[0]
        image = cv2.imread(img_path)
        label = cv2.imread(label_path, 0)   #读取标签使用单通道灰度图
        if self.is_label_divide:
            label = label // 255
        trans_train = Compose([
                ColorJitter(),
                # VerticalFlip(),
                # HorizontalFlip(),
                Resize(self.crop_size[0], self.crop_size[1]),
                Normalize(mean=self.img_mean, std=self.img_std)
            ])

        trans_val = Compose([
            Resize(self.crop_size[0], self.crop_size[1]),
            Normalize(mean=self.img_mean, std=self.img_std)
        ])
        aug_data = trans_train(image=image, mask=label) if self.train else trans_val(image=image, mask=label)
        x = aug_data["image"]
        target = aug_data["mask"]
        if image.ndim == 3:
            x = np.transpose(x, axes=[2, 0, 1])
        elif image.ndim == 2:
            x = np.expand_dims(x, axis=0)
        return torch.from_numpy(x), torch.from_numpy(target)