from typing import Any

import pytorch_lightning as pl
import torch
from torch import nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics import Accuracy, Dice


class LightningModel(pl.LightningModule):
    def __init__(self, model: nn.Module, learning_rate: float = 1e-3):
        super(LightningModel, self).__init__()
        self.learning_rate = learning_rate
        self.model = model

        self.train_accuracy = Accuracy(num_classes=1, average='micro', threshold=0.5, multiclass=False)
        self.val_accuracy = Accuracy(num_classes=1, average='micro', threshold=0.5, multiclass=False)

        self.test_accuracy = Accuracy(mdmc_average='samplewise')
        self.test_dice = Dice()

    def forward(self, x):
        return self.model(x)

    def get_backbone(self):
        return self.model

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                               factor=0.2, patience=3, threshold=0.001,
                                                               threshold_mode='abs')
        return {
            'optimizer': optimizer,
            'lr_scheduler': lr_scheduler,
            'monitor': 'val_loss'
        }

    def on_epoch_start(self) -> None:
        print("")

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self._loss(y_hat, y)
        self.train_accuracy(torch.flatten((y_hat > 0.5).type(torch.uint8), start_dim=1),
                            torch.flatten(y.type(torch.uint8), start_dim=1))

        self.log("train_loss", loss)
        self.log("train_accuracy", self.train_accuracy)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        val_loss = self._loss(y_hat, y)
        self.val_accuracy(torch.flatten((y_hat > 0.5).type(torch.uint8), start_dim=1),
                          torch.flatten(y.type(torch.uint8), start_dim=1))
        self.log("val_loss", val_loss, prog_bar=True)
        self.log("val_accuracy", self.val_accuracy, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        test_loss = self._loss(y_hat, y)
        self.test_accuracy(torch.flatten((y_hat > 0.5).type(torch.uint8), start_dim=1),
                           torch.flatten(y.type(torch.uint8), start_dim=1))

        self.log("test_loss", test_loss, prog_bar=True)
        self.log("test_accuracy", self.test_accuracy, prog_bar=True)

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        x, y = batch
        pred = self(x)
        pred_data = (pred > 0.5).type(torch.uint8)
        pred_data = pred_data.cpu().detach().numpy()

        return pred

    def _loss(self, y_pred, y_gt):
        return F.binary_cross_entropy(y_pred, y_gt)


def load_checkpoint(check_point, gpu, version: int = 2):
    if version == 1:
        from model_unet import Unet
    if version == 2:
        from model_unet_v2 import Unet
    elif version == 3:
        from model_unet_v3 import Unet

    if gpu:
        ckpt = torch.load(check_point)
    else:
        ckpt = torch.load(check_point, map_location=torch.device('cpu'))
    model = Unet(input_channels=3)
    lm = LightningModel(model)
    lm.load_state_dict(ckpt["state_dict"])
    return lm
