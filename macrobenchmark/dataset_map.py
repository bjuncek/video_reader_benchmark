import time
from typing import Callable, Optional

import numpy as np

import torch
from dataset_utils import DatasetConfig

from reader import Reader
from torchvision.datasets.folder import find_classes, make_dataset
from torchvision.datasets.vision import VisionDataset


class KineticsDataset(VisionDataset):
    def __init__(
        self,
        data_cfg: DatasetConfig,
        reader: Reader,
        transform: Optional[Callable] = None,
    ):
        # TODO: optional split options?
        self.cfg = data_cfg
        self.reader = reader
        super(KineticsDataset, self).__init__(data_cfg.root)
        self.transform = transform
        self.classes, class_to_idx = find_classes(data_cfg.root)
        self.samples = make_dataset(
            data_cfg.root, class_to_idx, data_cfg.extensions, is_valid_file=None
        )
        # modify number of samples based on the sampling stratedy
        self._get_sampling_info()

    def __len__(self):
        return len(self.samples)

    def _get_sampling_info(self):
        print(f"Preprocessing dataset: {self.cfg.sampling}...")
        print(f"\t len of dataset: {len(self.samples)}")
        start = time.time()
        newsamples = []
        for sample, target in self.samples:
            try:
                obj = self.reader(sample, self.cfg.video_cfg)
                duration = obj.get_duration()
                fps = obj.cfg.frame_rate
                clip_len_in_s = obj.cfg.clip_len / fps
                max_seek = max(
                    duration - clip_len_in_s - 0.5, 0
                )  # add a small buffer to avoid rounding errors
                if duration < clip_len_in_s:
                    print(f"\tSkipping: {sample} is too short to be sampled.")

                if self.cfg.sampling == "random":
                    newsamples.extend(
                        [(sample, target, max_seek, True)] * self.cfg.clip_multiplier
                    )
                elif self.cfg.sampling == "uniform":
                    tss = [
                        i.item()
                        for i in list(
                            torch.linspace(0, max_seek, steps=self.cfg.clip_multiplier)
                        )
                    ]
                    for start in tss:
                        newsamples.append((sample, target, start, False))
                else:
                    raise ValueError(f"Unknown sampling strategy: {self.cfg.sampling}")
            except Exception as e:
                print(f"\tSkipping: {sample} failed to be processed. : {e}")
        self.samples = newsamples
        end = time.time()
        print(f"\t len processed dataset: {len(self.samples)}")
        print(f"... took {end - start} seconds")

    def __getitem__(self, index: int):
        video, target, start, sample = self.samples[index]
        obj = self.reader(video, self.cfg.video_cfg)
        if sample:
            start = np.random.uniform(0, start)
        video = obj.read_video(start)
        if self.transform is not None:
            video = self.transform(video)
        target = torch.tensor(target)
        return video, target
