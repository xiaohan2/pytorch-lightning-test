"""
@File  :simple_net.py
@Author:chenyihan
@Date  :2023/12/4 15:26
@Desc  :
"""
import torch
import numpy as np
from torch import nn

class SimpleNet(nn.Module):
    def __init__(self, in_channel=3, out_channel=5, hid=128, layer_num=5):
        super().__init__()
        body = [nn.Conv2d(in_channel, hid, 3, padding=1),
                nn.ReLU()]
        for _ in range(layer_num-1):
            body.append(nn.Conv2d(hid, hid, 3, padding=1))
            body.append(nn.ReLU())

        self.body = nn.Sequential(*body)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(hid * 6 * 6, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, out_channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.body(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
