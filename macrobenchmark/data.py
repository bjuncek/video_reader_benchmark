from typing import Optional

import lightning.pytorch as pl

import torch

from dataset_utils import DatasetConfig
from dataset_map import KineticsDataset
from dataset_iterable import KineticsRandom, KineticsSequential


class KineticsDataModule(pl.LightningDataModule):
    def __init__(self, data_cfg: DatasetConfig, weights, reader, batch_size: int = 32):
        super().__init__()
        self.data_cfg = data_cfg
        self.reader = reader
        self.batch_size = batch_size
        self.transform = weights.transforms()

    def setup(self, stage: Optional[str] = None):
        if self.data_cfg.dataset_type == "mapstyle":
            self.train = KineticsDataset(
                self.data_cfg, self.reader, transform=self.transform
            )
        else:
            if self.data_cfg.sampling == "random":
                self.train = KineticsRandom(
                    self.data_cfg, self.reader, transform=self.transform
                )
            else:
                self.train = KineticsSequential(
                    self.data_cfg, self.reader, transform=self.transform
                )

    def train_dataloader(self):
        if self.data_cfg.dataset_type == "mapstyle":
            return torch.utils.data.DataLoader(
                self.train,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=8,
            )
        else:
            return torch.utils.data.DataLoader(
                self.train, batch_size=self.batch_size, num_workers=8
            )
