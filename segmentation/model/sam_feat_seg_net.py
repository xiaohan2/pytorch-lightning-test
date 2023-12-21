"""
@File  :sam_feat_seg_net.py
@Author:chenyihan
@Date  :2023/12/20 17:30
@Desc  :
"""
import torch.nn as nn
from autosam import sam_feat_seg_model_registry


class SamFeatSegNet(nn.Module):
    def __init__(
            self, sam_model_name, sam_checkpoint, num_classes
    ):
        super().__init__()
        self.model = sam_feat_seg_model_registry[sam_model_name](num_classes=num_classes, checkpoint=sam_checkpoint)

    def forward(self,
                x):
        out = self.model(x)
        return out
