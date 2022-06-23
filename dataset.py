import os
from typing import Optional

import cv2
import numpy as np
import pandas as pd
import torch.nn
from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
from torch.utils.data import Dataset, random_split, DataLoader
from torchvision import transforms


class MaskSegmentationTransform:
    def __call__(self, mask):
        mask = np.asarray(mask[..., 0], dtype=np.float32)
        mask[mask > 0] = 1
        return torch.from_numpy(mask[np.newaxis, ...])


def image_transform():
    return transforms.Compose([
        transforms.ToTensor()
    ])


def mask_transform():
    return transforms.Compose([
        MaskSegmentationTransform()
    ])


class GlasDataset(Dataset):
    def __init__(self, data_folder: str = 'augmented_dataset', train: bool = True):
        self.data_folder = data_folder
        self.image_transform = image_transform()
        self.mask_transform = mask_transform()

        df = pd.read_csv(os.path.join(data_folder, "grade.csv"))
        df = df.query(f"name.str.startswith('{'train' if train else 'test'}')")
        self.df = df

        self.images = {}

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, item_idx):
        item = self.df.iloc[item_idx]
        image_name = item['image']

        if image_name not in self.images:
            image = cv2.imread(os.path.join(self.data_folder, image_name))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            self.images[image_name] = image

        image = self.image_transform(self.images[image_name])

        mask_name = item['mask']

        if mask_name not in self.images:
            mask = cv2.imread(os.path.join(self.data_folder, mask_name))
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
            self.images[mask_name] = mask

        mask = self.mask_transform(self.images[mask_name])

        return image, mask


class GlassData(LightningDataModule):
    def __init__(self, data_folder: str, batch_size: int = 64, validation_split: float = 0.1, num_workers: int = 0):
        super(GlassData, self).__init__()
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.data_folder = data_folder
        self.num_workers = num_workers

        self.test_data = None
        self.val_split = None
        self.train_split = None
        self.training_data = None

    def setup(self, stage: Optional[str] = None) -> None:
        if stage in (None, 'fit'):
            self.training_data = GlasDataset(self.data_folder, train=True)
            dataset_length = len(self.training_data)
            val_size = int(self.validation_split * dataset_length)
            train_size = dataset_length - val_size
            self.train_split, self.val_split = random_split(self.training_data, [train_size, val_size])

        if stage in (None, 'test'):
            self.test_data = GlasDataset(self.data_folder, train=False)

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self.train_split, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.val_split, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.test_data, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)