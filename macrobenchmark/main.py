from torchvision.models.video import R3D_18_Weights
import lightning.pytorch as pl
import argparse


from model import BenchmarkModel
from data import KineticsDataModule, DatasetConfig
from reader import TVReader, PYAVReader

def main(args):
    _weights = R3D_18_Weights.DEFAULT

    datacfg = DatasetConfig(sampling=args.sampling, clip_multiplier=args.clip_multiplier)
    reader = PYAVReader

    trainer = pl.Trainer(max_epochs=1, profiler="simple", accelerator="gpu", devices=1)

    model = BenchmarkModel(_weights)
    datamodule = KineticsDataModule(datacfg, _weights, reader=reader, batch_size=2)

    trainer.fit(model, datamodule)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_type", type=str, choices=["iterable", "mapstyle"], default="mapstyle")
    parser.add_argument("--sampling", type=str, choices=["random", "uniform"], default="random")
    parser.add_argument("--profiler", type=str, default="simple")
    parser.add_argument("--reader", type=str, choices=["tv", "pyav", "ta"], default="gpu")
    parser.add_argument("--clip_multiplier", type=int, default=1)
    args = parser.parse_args()
    main(args)
