"""
@File  :infer.py
@Author:chenyihan
@Date  :2023/12/16 17:57
@Desc  :
"""
import cv2
from torch.nn import functional as F
from model.pid_net import PidNet
from albumentations import Compose, Resize, Normalize, ColorJitter
import torch
import numpy as np
from PIL import Image
import os

def inference(model, image):
    size = image.size()
    pred = model(image)
    pred = pred[1]

    pred = F.interpolate(
        input=pred, size=size[-2:],
        mode='bilinear', align_corners=True
    )
    return pred.exp()

def save_pred(preds, sv_path, origin_shape):
    preds = np.asarray(np.argmax(preds.detach().cpu().numpy(), axis=1), dtype=np.uint8)
    for i in range(preds.shape[0]):
        save_img = Image.fromarray(preds[i])
        save_img = save_img.resize((origin_shape[1], origin_shape[0]))
        save_img.save(sv_path)

if __name__ == '__main__':
    model = PidNet(num_classes=5)
    model.eval()
    device = torch.device("cuda:1" if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    weights = r'C:\Users\chenyihan\PycharmProjects\pytorch_lightning_test\segmentation\test.pt'
    state = torch.load(weights, map_location="cpu")
    model_state = state
    try:
        model.load_state_dict(model_state)
    except:
        from collections import OrderedDict

        new_state = OrderedDict()
        for k, v in model_state.items():
            new_state[k[7:]] = model_state[k]
        model.load_state_dict(new_state)
    image = cv2.imread(r'C:\Users\chenyihan\Desktop\S33_old\img_test\20.bmp')
    origin_shape = image.shape[:-1]
    trans_val = Compose([
        Resize(2400,3200),
        Normalize(mean=[0, 0, 0], std=[1, 1, 1])
    ])
    trans_image = trans_val(image=image)['image']
    trans_image = np.transpose(trans_image,axes=[2, 0, 1])
    input_tensor = torch.from_numpy(trans_image).unsqueeze(dim=0).to(device)
    pred = inference(model, input_tensor)
    save_pred(pred, 'test.png', origin_shape)