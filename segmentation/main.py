"""
@File  :main.py
@Author:chenyihan
@Date  :2023/12/4 15:05
@Desc  :
"""
import json
import os

import pytorch_lightning as pl
from argparse import ArgumentParser
from pytorch_lightning import Trainer
import pytorch_lightning.callbacks as plc
from pytorch_lightning.loggers import TensorBoardLogger
from model import MInterface
from data import DInterface
from utils import load_model_path_by_args
import torch.distributed as dist
from pytorch_lightning.strategies import DDPStrategy


def load_callbacks(args):
    callbacks = [plc.EarlyStopping(
        monitor='val_iou',
        mode='max',
        patience=1000,
        min_delta=0.0001
    ), plc.ModelCheckpoint(
        monitor='val_iou',
        filename='best-{epoch:02d}-{val_iou:.3f}',
        save_top_k=1,
        mode='max',
        save_last=True
    )]

    if args.lr_scheduler:
        callbacks.append(plc.LearningRateMonitor(
            logging_interval='epoch'))
    return callbacks


def main(args):
    pl.seed_everything(args.seed)
    load_path = load_model_path_by_args(args)
    data_module = DInterface(**vars(args))

    if load_path is None:
        model = MInterface(**vars(args))
    else:
        model = MInterface(**vars(args))
        args.ckpt_path = load_path

    # # If you want to change the logger's saving folder
    # logger = TensorBoardLogger(save_dir='kfold_log', name=args.log_dir)
    args.callbacks = load_callbacks(args)
    # args.logger = logger
    # 根据当前系统决定使用哪个backend..  PS:windows下分布式训练只只能用gloo，只有linux才支持nccl
    precision = '32-true'
    if args.mixed_precision:
        precision = '16-mixed'
    trainer = Trainer(limit_train_batches=100, max_epochs=args.epochs, devices=args.devices, log_every_n_steps=16,
                      callbacks=args.callbacks, precision=precision)
    if len(args.devices) > 1:
        platform_name = os.name
        strategy = DDPStrategy(process_group_backend='nccl', find_unused_parameters=True)
        if platform_name == 'nt':
            strategy = DDPStrategy(process_group_backend='gloo', find_unused_parameters=True)
        trainer = Trainer(limit_train_batches=100, max_epochs=args.epochs, devices=args.devices, log_every_n_steps=16,
                          callbacks=args.callbacks, strategy=strategy, precision=precision)
    trainer.fit(model, data_module)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--cfg', default="config.json", type=str)
    args = parser.parse_args()

    # 将json的配置参数转为args
    args_dict = vars(args)
    config_path = args.cfg
    with open(config_path, 'r', encoding='UTF-8') as f:
        json_dict = json.load(f)
        args_dict.update(json_dict)
    main(args)
