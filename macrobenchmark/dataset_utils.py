import os
from dataclasses import dataclass
from typing import Callable, Optional

import torch
from reader import Reader, VideoConfig
from torchvision import transforms as t
from torchvision.datasets.folder import make_dataset


def _find_classes(dir):
    classes = [d.name for d in os.scandir(dir) if d.is_dir()]
    classes.sort()
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    return classes, class_to_idx


def get_samples(root, extensions=(".mp4", ".avi")):
    _, class_to_idx = _find_classes(root)
    return make_dataset(root, class_to_idx, extensions=extensions)


@dataclass
class DatasetConfig:
    dataset_type: str = "mapstyle"
    root: str = "/home/bjuncek/data/kinetics/val/val"
    clip_multiplier: int = 1
    batch_size: int = 2
    sampling: str = "random"
    extensions: tuple = (".mp4", ".avi")
    video_cfg: VideoConfig = VideoConfig()

    def __post_init__(self):
        assert self.sampling in [
            "random",
            "uniform",
        ], "sampling must be either 'random' or 'uniform'"
        assert self.clip_multiplier > 0, "clip_multiplier must be greater than 0"


class KineticsIterDataset(torch.utils.data.IterableDataset):
    """Template class for iterable datasets."""

    def __init__(
        self,
        datasetcfg: DatasetConfig,
        reader: Reader,
        transform: Optional[Callable] = None,
        alpha: float = 0.1,
    ):
        super(KineticsIterDataset).__init__()

        self.reader = reader
        self.transform = transform

        self.samples = get_samples(datasetcfg.root)
        self.sampling = datasetcfg.sampling

        # allow for temporal jittering
        self.epoch_size = len(self.samples) * datasetcfg.clip_multiplier
        self.num_steps = self.epoch_size // len(self.samples)  # used for uniform

        self.cfg = datasetcfg
        self.alpha = alpha  # tollerance to avoid rounding errros with max seek time

    def __iter__(self):
        raise NotImplementedError
