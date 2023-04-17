import itertools
from dataclasses import dataclass
import math
import av
import torch
import torchvision
import warnings

@dataclass
class VideoConfig:
    clip_len: int = 16
    frame_rate: int = 30


class Reader(object):
    """Abstract base class for all readers."""

    def __init__(self, path, cfg, **kwargs):
        raise NotImplementedError

    def read_video(self, start):
        raise NotImplementedError

    def get_duration(self, path):
        raise NotImplementedError


class TVReader(Reader):
    """Reader for video files using torchvision.io."""

    def __init__(self, cfg: VideoConfig, **kwargs):
        torchvision.set_video_backend("pyav")
        super().__init__(cfg, **kwargs)

    def read_video(self, path, start):
        if self.path is not None and path != self.path:
            self.container = torchvision.io.VideoReader(path, "video")
            self.path = path
        frames = []
        for data in itertools.islice(self.container.seek(start), self.cfg.num_frames):
            frames.append(data["data"])
        frames.append(frames)
        return torch.stack(frames, 0)

    def get_duration(self, path):
        if (self.container is None or self.path is not None) or path != self.path:
            self.container = torchvision.io.VideoReader(path, "video")
            self.path = path

        return self.container.get_metadata()["video"]["duration"][0]


class PYAVReader(Reader):
    def __init__(self, path, cfg: VideoConfig, **kwargs):
        self.cfg = cfg
        self.container = av.open(path)
   
    def read_video(self, start):
        stream = self.container.streams.video[0]
        stream_name = {"video": 0}
        start_offset = int(math.floor(start * (1 / stream.time_base)))
        # check for crappy offsets
        seek_offset = max(start_offset - 5, 0)
        self.container.seek(seek_offset, any_frame=False, backward=True, stream=stream)

        frames = {}
        for _, frame in enumerate(self.container.decode(**stream_name)):
            if frame.pts >= start_offset:
                frames[frame.pts] = frame

            if len(frames) >= self.cfg.clip_len:
                break

        result = [frames[i] for i in sorted(frames)]
        if len(result) < self.cfg.clip_len:
            warnings.warn("Not enough frames found, padding with last frame")
            result = result + [result[-1]] * (self.cfg.clip_len - len(result))
        result = [torch.from_numpy(frame.to_rgb().to_ndarray()) for frame in result]
        ret = torch.stack(result, 0).permute(0, 3, 1, 2)
        return ret

    def get_duration(self):
        dur = self.container.streams.video[0].duration * self.container.streams.video[0].time_base
        return float(dur)
