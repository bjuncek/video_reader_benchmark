import lightning.pytorch as pl

import torch
import torch.nn.functional as F

from torchvision.models.video import r3d_18


class BenchmarkModel(pl.LightningModule):
    def __init__(self, weights):
        super().__init__()
        # Step 1: Initialize model with the best available weights
        self.model = r3d_18(weights=weights)
        self.loss = torch.nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x).softmax(0)
        loss = self.loss(y_hat, y)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)
