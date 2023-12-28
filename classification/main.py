"""
@File  :main.py
@Author:chenyihan
@Date  :2023/12/4 15:05
@Desc  :
"""

import pytorch_lightning as pl
from argparse import ArgumentParser
from pytorch_lightning import Trainer
import pytorch_lightning.callbacks as plc
from pytorch_lightning.loggers import TensorBoardLogger
import lightning
from model import MInterface
from data import DInterface
from utils import load_model_path_by_args
import json


def load_callbacks(args):
    callbacks = [plc.EarlyStopping(
        monitor='val_acc',
        mode='max',
        patience=50,
        min_delta=0.001
    ), plc.ModelCheckpoint(
        monitor='val_acc',
        filename='best-{epoch:02d}-{val_acc:.3f}',
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

    # # If you want to change the logger's saving folderer
    # logger = TensorBoardLogger(save_dir='kfold_log', name=args.log_dir)
    args.callbacks = load_callbacks(args)
    # args.logger = logger

    trainer = Trainer(limit_train_batches=100, max_epochs=args.epochs, devices=args.devices, log_every_n_steps=16, callbacks=args.callbacks)
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