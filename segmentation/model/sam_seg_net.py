"""
@File  :sam_seg_net.py
@Author:chenyihan
@Date  :2023/12/20 18:15
@Desc  :
"""
import torch.nn as nn
from autosam import sam_seg_model_registry


class SamSegNet(nn.Module):
    def __init__(
            self, sam_model_name, sam_checkpoint, num_classes
    ):
        super().__init__()
        self.model = sam_seg_model_registry[sam_model_name](num_classes=num_classes, checkpoint=sam_checkpoint)

    def forward(self,
                x):
        out = self.model(x)
        return out