"""
@File  :utils.py
@Author:chenyihan
@Date  :2023/12/4 15:05
@Desc  :
"""
import os
from collections import OrderedDict

import timm
import torch
from pathlib2 import Path
import cv2
import numpy as np

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


def export_script(model_path, export_path, num_classes=2):
    model = timm.create_model('mobilevit_s.cvnets_in1k', pretrained=False, num_classes=num_classes)
    model.eval()
    state = torch.load(model_path)['state_dict']
    new_state = OrderedDict()
    for k, v in state.items():
        new_state[k[6:]] = state[k]
    model.load_state_dict(new_state)
    example_input = torch.randn(1, 3, 768, 768)

    # 进行跟踪tracing
    traced_model = torch.jit.trace(model, example_input)
    traced_model.save(export_path)


if __name__ == '__main__':
    # root_path = "dataset/2016tiehuan"
    # generate_txt(root_path)
    model_path = r"C:\Users\chenyihan\Desktop\model_class\wangyin_val_acc=1.000.ckpt"
    export_path = "wangyin.pt"
    export_script(model_path,export_path)