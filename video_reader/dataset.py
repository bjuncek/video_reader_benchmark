import os
import random

import torch
from torchvision.datasets.folder import make_dataset
from torchvision import transforms as t

from video import Video


def _find_classes(dir):
    classes = [d.name for d in os.scandir(dir) if d.is_dir()]
    classes.sort()
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    return classes, class_to_idx

def read_video_frames(vid, start=0, nframes=1, height=-1, width=-1, read_video=True, read_audio=False, from_keyframes=True):
    if not isinstance(vid, Video):
        vid = Video(path)    
    # safety checks, streams
    stream_types = [x['type'] for x in vid.available_streams]
    if read_video:
        assert "video" in stream_types
    if read_audio:
        assert "audio" in stream_types
    
    # get video_transform to apply per frame
    # should save on memory
    transforms = [t.ToTensor()]
    if width > 0 and height>0:
        transforms.insert(0, t.Resize((height, width), interpolation=2))
        transforms.insert(0, t.ToPILImage())     
    frame_transform = t.Compose(transforms)
    
    current_pts = start
    if read_video:
        video_frames = [] # video frame buffer 
    if read_audio:
        audio_frames = [] # audio frame buffer
            
    # this should get us close to the actual starting point we want
    vid.seek(start, stream="video")
    while len(video_frames) < nframes:
        frame, current_pts, stream_t = vid.next("video")
        if from_keyframes:
            video_frames.append(frame_transform(frame))
        else:
            frame, current_pts, stream_t = vid.next("video")
            if current_pts >= start:
                video_frames.append(frame_transform(frame))
    
    output = {'video': torch.stack(video_frames, 0) if read_video else torch.empty(0)}
    return output

class VideoDataset(torch.utils.data.IterableDataset):
    def __init__(self, root, clip_len=16, shuffle=True, bs_multiplier=5, sampling='random', alpha=0.2, height=-1, width=-1):
        super(VideoDataset).__init__()
        # safety checks
        assert isinstance(bs_multiplier, int) and bs_multiplier >= 1
        assert sampling in ["random", "uniform"]
        
        self.root = root
        self.clip_len = clip_len
        self.height=height
        self.width=width
        self.alpha = alpha  #  hack to ensure readin is correct
        self._build_dataset(bs_multiplier, sampling)  
    
    def _build_dataset(self, bs_multiplier, sampling):
        _, ctidx = _find_classes(self.root)
        samples = make_dataset(self.root, ctidx, extensions=(".mp4", ".avi"))
        self.samples = []
        for sample in samples:
            path, target = sample
            vid = Video(path, debug=False)
            max_seek = vid.duration - (self.clip_len / vid.fps + self.alpha)
            if sampling == "random":
                tss = sorted([random.uniform(0., max_seek) for _ in range(bs_multiplier)])
            else:
                step = max(length // self.max_clips_per_video, 1)
                tss = [i.item() for i in list(torch.linspace(0, max_seek, steps=bs_multiplier))]
            
            for ts in tss:
                self.samples.append((path, target, ts))
    
    def _get_sample(self, sample):
        path, target, ts = sample
        vid = Video(path, debug=False)
        sample = read_video_frames(vid, start=ts, nframes=self.clip_len, height=self.height, width=self.width)
        sample['target'] = target
        return sample
        
    
    def __iter__(self):
        return iter([self._get_sample(sample) for sample in self.samples])
