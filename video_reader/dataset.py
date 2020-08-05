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

def get_samples(root, extensions=(".mp4", ".avi")):
    _, class_to_idx = _find_classes(root)
    return make_dataset(root, class_to_idx, extensions=extensions)

class GenericVideoDataset(torch.utils.data.IterableDataset):
    def __init__(self, root, epoch_size=None, frame_transform=None, video_transform=None, clip_len=16, shuffle=True, from_keyframes=True, alpha=0.1):
        super(GenericVideoDataset).__init__()
        
        self.samples = get_samples(root)
         
        # allow for temporal jittering
        if epoch_size is None:
            epoch_size = len(self.samples)
        self.epoch_size = epoch_size
        
        self.clip_len = clip_len  # length of a clip in frames
        self.frame_transform = frame_transform  # transform for every frame individually
        self.video_transform = video_transform # transform on a video sequence
        # FIXME: maybe remove
        self.alpha = alpha # tollerance to avoid rounding errros with max seek time
        self.from_keyframes = from_keyframes  # if true, only decode from the keyframes

    def __iter__(self):
        for i in range(self.epoch_size):
            # get random sample
            path, target = random.choice(self.samples)
            # get video object
            vid = Video(path, debug=False)
            video_frames = [] # video frame buffer 
            # seek and return frames
            max_seek = vid.metadata[vid.current_stream]['duration'] - (self.clip_len / vid.metadata[vid.current_stream]['fps'] + self.alpha)
            start = random.uniform(0., max_seek)
            vid.seek(start, stream="video", any_frame=self.from_keyframes)
            while len(video_frames) < self.clip_len:
                frame, current_pts, stream_t = vid.next("video")
                video_frames.append(self.frame_transform(frame))
            # stack it into a tensor
            video = torch.stack(video_frames, 0)
            if self.video_transform:
                video = self.video_transform(video)
            output = {
                'path': path,
                'video': video,
                'target': target,
                'start': start,
                'end': current_pts}
            yield output