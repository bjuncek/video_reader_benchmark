import random

import torch
from dataset_utils import KineticsIterDataset


class KineticsRandom(KineticsIterDataset):
    def __init__(self, *args, **kwargs):
        super(KineticsRandom, self).__init__(*args, **kwargs)

    def __iter__(self):
        for i in range(self.epoch_size):
            # get random sample
            path, target = random.choice(self.samples)
            # get video object
            vid = self.reader(path, self.cfg.video_cfg)

            # seek and return frames
            duration = vid.get_duration()
            fps = vid.cfg.frame_rate
            clip_len_in_s = vid.cfg.clip_len / fps
            max_seek = duration - clip_len_in_s - self.alpha
            start = random.uniform(0.0, max_seek)

            video = vid.read_video(start)
            if self.transform:
                video = self.transform(video)
            yield video, torch.tensor(target)


class KineticsSequential(KineticsIterDataset):
    def __init__(self, *args, **kwargs):
        super(KineticsSequential, self).__init__(*args, **kwargs)

    def __iter__(self):
        for i in range(len(self.samples)):
            # get random sample
            path, target = self.samples[i]
            # get video object
            vid = self.reader(path, self.cfg.video_cfg)

            # seek and return frames
            duration = vid.get_duration()
            fps = vid.cfg.frame_rate
            clip_len_in_s = vid.cfg.clip_len / fps
            max_seek = duration - clip_len_in_s - self.alpha

            tss = [
                i.item()
                for i in list(torch.linspace(0, max_seek, steps=self.num_steps))
            ]
            for start in tss:
                print(i, duration, start)
                video = vid.read_video(start)
                if self.transform:
                    video = self.transform(video)
                yield video, torch.tensor(target)
