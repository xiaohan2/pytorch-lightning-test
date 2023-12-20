"""
@File  :utils.py
@Author:chenyihan
@Date  :2023/12/4 15:05
@Desc  :
"""
import os
from pathlib2 import Path
import cv2
import numpy as np
import torch.nn as nn
import torch
from torch.nn import functional as F
from model.prompt_sam import PromptSam
from collections import OrderedDict

def load_model_path(root=None, version=None, v_num=None, best=False):
    """ When best = True, return the best model's path in a directory
        by selecting the best model with largest epoch. If not, return
        the last model saved. You must provide at least one of the
        first three args.
    Args:
        root: The root directory of checkpoints. It can also be a
            model ckpt file. Then the function will return it.
        version: The name of the version you are going to load.
        v_num: The version's number that you are going to load.
        best: Whether return the best model.
    """

    def sort_by_epoch(path):
        name = path.stem
        epoch = int(name.split('-')[1].split('=')[1])
        return epoch

    def generate_root():
        if root is not None:
            return root
        elif version is not None:
            return str(Path('lightning_logs', version, 'checkpoints'))
        else:
            return str(Path('lightning_logs', f'version_{v_num}', 'checkpoints'))

    if root == version == v_num == None:
        return None

    root = generate_root()
    if Path(root).is_file():
        return root
    if best:
        files = [i for i in list(Path(root).iterdir()) if i.stem.startswith('best')]
        files.sort(key=sort_by_epoch, reverse=True)
        res = str(files[0])
    else:
        res = str(Path(root) / 'last.ckpt')
    return res


def load_model_path_by_args(args):
    return load_model_path(root=args.load_dir, version=args.load_ver, v_num=args.load_v_num)


def write_to_file(file_path, data):
    with open(file_path, 'w', encoding='utf-8', newline='\r\n') as f:
        for item in data:
            f.write(item + "\n")


"""
生成所有图像的路径和其对应的标签，用于图像分类任务
"""


def generate_txt(root_path):
    txt_path = os.path.join(root_path, "files.txt")
    all_infos = []
    for root, dirs, files in os.walk(root_path):
        # print("root",root)
        # print("dirs",dirs)
        # print("files",files)
        for i, dir in enumerate(dirs):
            img_dir = os.path.join(root, dir)
            if os.path.isfile(img_dir):
                continue
            img_file_paths = os.listdir(img_dir)
            infos = [os.path.join(img_dir, file) + " " + str(i) for file in img_file_paths]
            all_infos.extend(infos)
    write_to_file(txt_path, all_infos)


def get_confusion_matrix(label, pred, size, num_class, ignore=-1):
    """
    Calcute the confusion matrix by given label and pred
    """
    output = pred.cpu().numpy().transpose(0, 2, 3, 1)
    seg_pred = np.asarray(np.argmax(output, axis=3), dtype=np.uint8)
    seg_gt = np.asarray(
        label.cpu().numpy()[:, :size[-2], :size[-1]], dtype=int)

    ignore_index = seg_gt != ignore
    seg_gt = seg_gt[ignore_index]
    seg_pred = seg_pred[ignore_index]

    index = (seg_gt * num_class + seg_pred).astype('int32')
    label_count = np.bincount(index)
    confusion_matrix = np.zeros((num_class, num_class))

    for i_label in range(num_class):
        for i_pred in range(num_class):
            cur_index = i_label * num_class + i_pred
            if cur_index < len(label_count):
                confusion_matrix[i_label,
                i_pred] = label_count[cur_index]
    return confusion_matrix


def weighted_bce(bd_pre, target):
    n, c, h, w = bd_pre.size()
    log_p = bd_pre.permute(0, 2, 3, 1).contiguous().view(1, -1)
    target_t = target.view(1, -1)

    pos_index = (target_t == 1)
    neg_index = (target_t == 0)

    weight = torch.zeros_like(log_p, dtype=torch.float32)
    pos_num = pos_index.sum()
    neg_num = neg_index.sum()
    sum_num = pos_num + neg_num
    weight[pos_index] = neg_num * 1.0 / sum_num
    weight[neg_index] = pos_num * 1.0 / sum_num

    loss = F.binary_cross_entropy_with_logits(log_p, target_t, weight, reduction='mean')

    return loss


class BondaryLoss(nn.Module):
    def __init__(self, coeff_bce=20.0):
        super(BondaryLoss, self).__init__()
        self.coeff_bce = coeff_bce

    def forward(self, bd_pre, bd_gt):
        bce_loss = self.coeff_bce * weighted_bce(bd_pre, bd_gt)
        loss = bce_loss

        return loss


class OhemCrossEntropy(nn.Module):
    def __init__(self, ignore_label=-1, thres=0.7,
                 min_kept=100000, weight=None):
        super(OhemCrossEntropy, self).__init__()
        self.thresh = thres
        self.min_kept = max(1, min_kept)
        self.ignore_label = ignore_label
        self.criterion = nn.CrossEntropyLoss(
            weight=weight,
            ignore_index=ignore_label,
            reduction='none'
        )

    def _ce_forward(self, score, target):
        ph, pw = score.size(2), score.size(3)
        h, w = target.size(1), target.size(2)
        if ph != h or pw != w:
            score = F.interpolate(input=score, size=(
                h, w), mode='nearest')

        loss = self.criterion(score, target)

        return loss

    def _ohem_forward(self, score, target, **kwargs):
        ph, pw = score.size(2), score.size(3)
        h, w = target.size(1), target.size(2)
        if ph != h or pw != w:
            score = F.interpolate(input=score, size=(
                h, w), mode='nearest')
        pred = F.softmax(score, dim=1)
        pixel_losses = self.criterion(score, target).contiguous().view(-1)
        mask = target.contiguous().view(-1) != self.ignore_label

        tmp_target = target.clone()
        tmp_target[tmp_target == self.ignore_label] = 0
        pred = pred.gather(1, tmp_target.unsqueeze(1))
        pred, ind = pred.contiguous().view(-1, )[mask].contiguous().sort()
        min_value = pred[min(self.min_kept, pred.numel() - 1)]
        threshold = max(min_value, self.thresh)

        pixel_losses = pixel_losses[mask][ind]
        pixel_losses = pixel_losses[pred < threshold]
        return pixel_losses.mean()

    def forward(self, score, target):
        weights = [1, 1, 1]
        assert len(weights) == len(score)
        functions = [self._ce_forward] * \
                    (len(weights) - 1) + [self._ohem_forward]
        # print("loss weight : ",weights, len(score), functions)
        loss = torch.unsqueeze(sum([
            w * func(x, target)
            for (w, x, func) in zip(weights, score, functions)
        ]), 0).mean()
        return loss


def calc_iou(result_contours, label_contours):
    assert len(result_contours) == len(label_contours)

    total_iou = 0
    for i in range(len(result_contours)):
        # ignore the background
        if i == 0:
            continue

        # ignore the neibi
        # if i == 9 or i == 10:
        #     continue

        result_mask = result_contours[i]
        label_mask = label_contours[i]
        intersection = np.logical_and(result_mask, label_mask)
        union = np.logical_or(result_mask, label_mask)
        # plt.imshow(label_mask, cmap='gray')
        # plt.show()
        # plt.imshow(result_mask, cmap='gray')
        # plt.show()

        # the empty situation
        if np.all(union == 0):
            iou = 1
        else:
            iou = np.sum(intersection) / np.sum(union)
        total_iou += iou

    # -1：ignore the background
    return total_iou / (len(result_contours) - 1)


def checkpoint2model(ckpt_file, save_path):
    """
    将ckpt文件转为只包含模型参数的pt文件
    :param ckpt_file:   pytorchlightning存的ckpt文件的路径
    :param save_path:   保存模型参数的pt文件路径
    """
    checkpoint = torch.load(ckpt_file)
    # model.load_state_dict(checkpoint['state_dict'])
    state = checkpoint['state_dict']
    new_state = OrderedDict()
    for k, v in state.items():
        new_state[k[6:]] = state[k]
    torch.save(new_state, save_path)


if __name__ == '__main__':

    root_path = "dataset/flower_photos"
    ckpt_file = r"C:\Users\chenyihan\PycharmProjects\best-epoch=318-val_iou=0.864.ckpt"
    weight_path = "test.pt"
    # generate_txt(root_path)
    checkpoint2model(ckpt_file, weight_path)
    # test the valid of model file
    # model = PromptSam('vit_t', "C:\\Users\\chenyihan\\Downloads\\mobile_sam.pt", 8)
    # state = torch.load(weight_path, map_location="cpu")
    # try:
    #     model.load_state_dict(state)
    # except:
    #     new_state = OrderedDict()
    #     for k, v in state.items():
    #         new_state[k[7:]] = state[k]         # PS:由于ckpt里存的'state_dict'的key多了个前缀"model.",所以从第六个字符开始
    #     model.load_state_dict(new_state)
    # print(model)