"""
@File  :classify_data.py
@Author:chenyihan
@Date  :2023/12/4 16:32
@Desc  用于图像分类的数据集接口:
"""
import torch
import os.path as op
import numpy as np
import torch.utils.data as data
import cv2
from torchvision import transforms
from albumentations import Compose, Resize, Normalize, ColorJitter, VerticalFlip, HorizontalFlip
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split


class ClassifyData(data.Dataset):
    def __init__(self, data_dir=r'../dataset/flower_photos',
                 class_num=5,
                 train=True,
                 no_augment=True,
                 input_size=[224, 224],
                 aug_prob=0.5,
                 img_mean=[0.485, 0.456, 0.406],
                 img_std=[0.229, 0.224, 0.225]):
        # Set all input args as attributes
        self.__dict__.update(locals())
        self.aug = train and not no_augment
        self.check_files()

    def check_files(self):
        # This part is the core code block for load your own dataset.
        # You can choose to scan a folder, or load a file list pickle
        # file, or any other formats. The only thing you need to gua-
        # rantee is the `self.path_list` must be given a valid value.
        file_list_path = op.join(self.data_dir, 'files.txt')
        with open(file_list_path, 'r') as f:
            file_list = [file.strip("\n") for file in f.readlines()]
        fl_train, fl_val = train_test_split(
            file_list, test_size=0.2, random_state=2023)
        self.path_list = fl_train if self.train else fl_val

    def __len__(self):
        return len(self.path_list)

    def to_one_hot(self, idx):
        out = np.zeros(self.class_num, dtype=float)
        out[idx] = 1
        return out

    def __getitem__(self, idx):
        img_path = self.path_list[idx].split(" ")[0]
        label_index = self.path_list[idx].split(' ')[-1]
        filename = op.splitext(op.basename(img_path))[0]
        img = cv2.imread(img_path)
        labels = self.to_one_hot(int(label_index))
        labels = torch.from_numpy(labels).float()

        trans = Compose([
            ColorJitter(),
            VerticalFlip(),
            HorizontalFlip(),
            Resize(self.input_size[0], self.input_size[1]),
            Normalize(mean=self.img_mean, std=self.img_std),
            ToTensorV2(),
        ]
        ) if self.train else Compose(
            [
                Resize(self.input_size[0], self.input_size[1]),
                Normalize(mean=self.img_mean, std=self.img_std),
                ToTensorV2(),
            ]
        )

        img_tensor = trans(image=img)['image']
        return img_tensor, labels
