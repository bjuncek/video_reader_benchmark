from typing import Optional

import lightning.pytorch as pl

import torch

from dataset_utils import DatasetConfig


class KineticsDataModule(pl.LightningDataModule):
    def __init__(self, data_cfg: DatasetConfig, weights, reader, batch_size: int = 32):
        super().__init__()
        self.data_cfg = data_cfg
        self.reader = reader
        self.batch_size = batch_size
        self.transform = weights.transforms()

    def setup(self, stage: Optional[str] = None):
        if self.data_cfg.dataset_type == "mapstyle":
            from dataset_map import KineticsDataset

            self.train = KineticsDataset(
                self.data_cfg, self.reader, transform=self.transform
            )
        else:
            if self.data_cfg.sampling == "uniform":
                from dataset_iterable import KineticsSequential

                self.train = KineticsSequential(
                    self.data_cfg, self.reader, transform=self.transform
                )
            elif self.data_cfg.sampling == "random":
                from dataset_iterable import KineticsRandom

                self.train = KineticsRandom(
                    self.data_cfg, self.reader, transform=self.transform
                )
            else:
                raise ValueError("sampling must be either 'random' or 'uniform'")

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train, batch_size=self.batch_size, shuffle=True, num_workers=8
        )
