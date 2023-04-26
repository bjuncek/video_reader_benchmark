import itertools
from dataclasses import dataclass
import math
import av
import torch
import torchvision
import torchaudio
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

    def __init__(self, path, cfg: VideoConfig, **kwargs):
        torchvision.set_video_backend("pyav")
        self.container = torchvision.io.VideoReader(path, "video")
        self.cfg = cfg

    def read_video(self, start):
        result = []
        for data in itertools.islice(self.container.seek(start), self.cfg.clip_len):
            result.append(data["data"])

        if len(result) < self.cfg.clip_len:
            warnings.warn("Not enough frames found, padding with last frame")
            result = result + [result[-1]] * (self.cfg.clip_len - len(result))
        if len(result) > self.cfg.clip_len:
            result = result[: self.cfg.clip_len]
        return torch.stack(result, 0)

    def get_duration(self):
        return self.container.get_metadata()["video"]["duration"][0]


class TAReader(Reader):
    """Reader for video files using torchaudio.io.
    NOT TESTED"""

    def __init__(self, path, cfg: VideoConfig, **kwargs):
        self.container = torchaudio.io.StreamReader(path)
        self.container.add_basic_video_stream(
            frames_per_chunk=1,
            format="rgb24",
            frame_rate=cfg.frame_rate,
        )
        self.cfg = cfg

    def read_video(self, start):
        result = []
        curr = 0
        counter = 0
        for chunks in self.container.stream():
            if len(result) < self.cfg.clip_len:
                curr = counter / self.cfg.frame_rate
                if curr >= start:
                    result.append(chunks[0])
                counter += 1
        if len(result) < self.cfg.clip_len:
            warnings.warn("Not enough frames found, padding with last frame")
            result = result + [result[-1]] * (self.cfg.clip_len - len(result))
        if len(result) > self.cfg.clip_len:
            result = result[: self.cfg.clip_len]
        return torch.stack(result, 0)

    def get_duration(self):
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

            if len(frames) > self.cfg.clip_len:
                break

        result = [frames[i] for i in sorted(frames)]
        if len(result) < self.cfg.clip_len:
            warnings.warn("Not enough frames found, padding with last frame")
            result = result + [result[-1]] * (self.cfg.clip_len - len(result))
        if len(result) > self.cfg.clip_len:
            result = result[: self.cfg.clip_len]
        result = [torch.from_numpy(frame.to_rgb().to_ndarray()) for frame in result]
        ret = torch.stack(result, 0).permute(0, 3, 1, 2)
        return ret

    def get_duration(self):
        dur = self.container.streams.video[0].duration * self.container.streams.video[0].time_base
        return float(dur)
