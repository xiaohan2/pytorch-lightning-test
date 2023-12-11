"""
@File  :model_interface.py
@Author:chenyihan
@Date  :2023/12/4 15:21
@Desc  :
"""
import inspect
import torch
import importlib
from torch.nn import functional as F
import torch.optim.lr_scheduler as lrs
import pytorch_lightning as pl
from utils import get_confusion_matrix
import numpy as np
import segmentation_models_pytorch as smp


class MInterface(pl.LightningModule):
    def __init__(self, model_name, loss, lr, **kargs):
        super().__init__()
        self.save_hyperparameters()
        self.load_model()
        self.configure_loss()

    def forward(self, img):
        return self.model(img)

    def training_step(self, batch, batch_idx):
        img, labels = batch
        labels = labels.long()
        out = self(img)
        if self.hparams["model_name"] == "pp_lite_seg":
            loss = sum([self.loss_function(x, labels) for x in out])
        else:
            loss = self.loss_function(out, labels)
        self.log('loss', loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=self.hparams["batch_size"])
        return loss

    def validation_step(self, batch, batch_idx):
        img, labels = batch
        labels = labels.long()
        out = self(img)
        mean_iou = 0
        if self.hparams["model_name"] == "pp_lite_seg":
            confusion_matrix = np.zeros(
                (self.hparams["num_classes"], self.hparams["num_classes"], 3))
            loss = sum([self.loss_function(x, labels) for x in out])
            size = labels.size()
            if not isinstance(out, (list, tuple)):
                out = [out]
            for i, x in enumerate(out):
                x = F.interpolate(
                    input=x, size=size[-2:],
                    mode='nearest'
                )
                confusion_matrix[..., i] += get_confusion_matrix(
                    labels,
                    x,
                    size,
                    self.hparams["num_classes"]
                )
            for i in range(3):
                pos = confusion_matrix[..., i].sum(1)
                res = confusion_matrix[..., i].sum(0)
                tp = np.diag(confusion_matrix[..., i])
                iou_array = (tp / np.maximum(1.0, pos + res - tp))
                iou = iou_array.mean()
                mean_iou += iou
            mean_iou /= 3
        else:
            loss = self.loss_function(out, labels)
            out = torch.softmax(out, dim=1)
            confusion_matrix = get_confusion_matrix(labels, out, labels.size(), self.hparams["num_classes"])
            pos = confusion_matrix.sum(1)
            res = confusion_matrix.sum(0)
            tp = np.diag(confusion_matrix)
            iou_array = (tp / np.maximum(1.0, pos + res - tp))
            mean_iou = iou_array.mean()
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=self.hparams["batch_size"],
                 sync_dist=len(self.hparams["devices"]) > 1)
        self.log('val_iou', mean_iou,
                 on_step=False, on_epoch=True, prog_bar=True, batch_size=self.hparams["batch_size"],
                 sync_dist=len(self.hparams["devices"]) > 1)

        return loss

    def test_step(self, batch, batch_idx):
        # Here we just reuse the validation_step for testing
        return self.validation_step(batch, batch_idx)

    def configure_optimizers(self):
        if hasattr(self.hparams, 'weight_decay'):
            weight_decay = self.hparams.weight_decay
        else:
            weight_decay = 0
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.hparams.lr, weight_decay=weight_decay)

        if self.hparams.lr_scheduler is None:
            return optimizer
        else:
            if self.hparams.lr_scheduler == 'step':
                scheduler = lrs.StepLR(optimizer,
                                       step_size=self.hparams.lr_decay_steps,
                                       gamma=self.hparams.lr_decay_rate)
            elif self.hparams.lr_scheduler == 'cosine':
                scheduler = lrs.CosineAnnealingLR(optimizer,
                                                  T_max=self.hparams.lr_decay_steps,
                                                  eta_min=self.hparams.lr_decay_min_lr)
            else:
                raise ValueError('Invalid lr_scheduler type!')
            return [optimizer], [scheduler]

    def configure_loss(self):
        loss = self.hparams.loss.lower()
        if loss == 'mse':
            self.loss_function = F.mse_loss
        elif loss == 'l1':
            self.loss_function = F.l1_loss
        elif loss == 'bce':
            self.loss_function = F.binary_cross_entropy
        elif loss == 'ce':
            self.loss_function = F.cross_entropy
        else:
            raise ValueError("Invalid Loss Type!")

    def load_model(self):
        name = self.hparams.model_name
        # Change the `snake_case.py` file name to `CamelCase` class name.
        # Please always name your model file name as `snake_case.py` and
        # class name corresponding `CamelCase`.
        camel_name = ''.join([i.capitalize() for i in name.split('_')])
        try:
            Model = getattr(importlib.import_module(
                '.' + name, package=__package__), camel_name)
        except:
            raise ValueError(
                f'Invalid Module File Name or Invalid Class Name {name}.{camel_name}!')
        self.model = self.instancialize(Model)

        # 按以下方式使用segmentation_models_pytorch中的模型
        # self.model = smp.Unet(
        #     encoder_name="resnet34",  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        #     encoder_weights="imagenet",  # use `imagenet` pre-trained weights for encoder initialization
        #     in_channels=3,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        #     classes=2,  # model output channels (number of classes in your dataset)
        # )

    def instancialize(self, Model, **other_args):
        """ Instancialize a model using the corresponding parameters
            from self.hparams dictionary. You can also input any args
            to overwrite the corresponding value in self.hparams.
        """
        class_args = list(inspect.signature(Model.__init__).parameters.keys())[1:]
        inkeys = self.hparams.keys()
        args1 = {}
        for arg in class_args:
            if arg in inkeys:
                args1[arg] = getattr(self.hparams, arg)
        args1.update(other_args)
        return Model(**args1)
