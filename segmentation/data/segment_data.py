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
                 num_classes=2,
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
        self.need_edge = config["need_edge"]
        self.edge_size = config["edge_size"]

    def __len__(self):
        return len(self.path_list)

    def __getitem__(self, idx):
        img_path, label_path = self.path_list[idx]
        img_path, label_path = os.path.join(self.data_dir, img_path), os.path.join(self.data_dir, label_path)
        filename = op.splitext(op.basename(img_path))[0]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
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
        # 获取边缘
        y_k_size = 6
        x_k_size = 6
        edge = cv2.Canny(target, 0.1, 0.2)
        kernel = np.ones((self.edge_size, self.edge_size), np.uint8)
        edge = edge[y_k_size:-y_k_size, x_k_size:-x_k_size]
        edge = np.pad(edge, ((y_k_size, y_k_size), (x_k_size, x_k_size)), mode='constant')
        edge = (cv2.dilate(edge, kernel, iterations=1) > 50) * 1.0

        # 产品分割时对标签进行处理
        masks = []
        for i in range(self.num_classes):
            temp = np.where(target == i, 1, 0)
            masks.append(temp)

        """
        2016标签的处理方式
        """
        # label已经左右的标签合的情况
        masks[1] = np.bitwise_or(masks[1], np.bitwise_or(masks[2], masks[3]))
        masks[4] = np.bitwise_or(masks[4], masks[5])
        masks[6] = np.bitwise_or(masks[6], masks[7])

        # # 基座不扣
        # masks[1] = np.bitwise_or(masks[1], np.bitwise_or(masks[2], masks[3]))
        # # merge left and right
        # masks[4] = np.bitwise_or(masks[4], masks[6])
        # masks[5] = np.bitwise_or(masks[5], masks[7])
        # masks[4] = np.bitwise_or(masks[4], masks[5])
        # masks[5] = np.bitwise_or(masks[6], masks[7])
        # # masks[6] = masks[8]
        # # masks[7] = masks[9]
        # 孤岛不扣
        # masks[6] = np.bitwise_or(masks[8], masks[9])
        # masks[7] = masks[9]
        # masks = masks[:8]

        """
        238标签的处理方式
        """
        # masks[8] = np.bitwise_or(masks[8], masks[11])
        # masks[9] = np.bitwise_or(masks[9], masks[12])
        # masks[10] = np.bitwise_or(masks[10], masks[13])
        # masks[5] = np.bitwise_or(masks[5], masks[10])
        # masks[4] = np.bitwise_or(np.bitwise_or(masks[4], masks[5]), np.bitwise_or(masks[8], masks[9]))
        # masks = masks[1:]
        target = np.array(masks)

        if image.ndim == 3:
            x = np.transpose(x, axes=[2, 0, 1])
        elif image.ndim == 2:
            x = np.expand_dims(x, axis=0)
        if self.need_edge:
            return torch.from_numpy(x), torch.from_numpy(target), torch.from_numpy(edge)
        return torch.from_numpy(x), torch.from_numpy(target)
