import argparse

import lightning.pytorch as pl
from dataset_utils import DatasetConfig
from data import KineticsDataModule

from model import BenchmarkModel
from reader import PYAVReader
from torchvision.models.video import R3D_18_Weights


def main(args):
    _weights = R3D_18_Weights.DEFAULT

    datacfg = DatasetConfig(
        dataset_type=args.dataset_type,
        sampling=args.sampling,
        clip_multiplier=args.clip_multiplier,
        batch_size=args.batch_size,
    )
    reader = PYAVReader

    trainer = pl.Trainer(max_epochs=1, devices=1)

    model = BenchmarkModel(_weights)
    datamodule = KineticsDataModule(datacfg, _weights, reader=reader)

    trainer.fit(model, datamodule)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_type", type=str, choices=["iterable", "mapstyle"], default="mapstyle"
    )
    parser.add_argument(
        "--sampling", type=str, choices=["random", "uniform"], default="random"
    )
    parser.add_argument("--profiler", type=str, default="simple")
    parser.add_argument(
        "--reader", type=str, choices=["tv", "pyav", "ta"], default="pyav"
    )
    parser.add_argument("--clip_multiplier", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    args = parser.parse_args()
    main(args)
