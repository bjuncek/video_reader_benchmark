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

class KFDataset(torch.utils.data.IterableDataset):
    def __init__(self, root, epoch_size=None, transform=None, clip_len=16):
        super(KFDataset).__init__()
        
        self.samples = get_samples(root)
         
        # allow for temporal jittering
        if epoch_size is None:
            epoch_size = len(self.samples)
        self.epoch_size = epoch_size
        
        self.clip_len = clip_len  # length of a clip in frames
        self.transform = frame_transform  # transform for every frame individually
    
    def __iter__(self):
        for i in range(self.epoch_size):
            path, target = random.choice(self.samples)
            # get video object
            vid = Video(path, debug=False)
            if vid.metadata[vid.current_stream]['duration'] > 0:
                start = random.uniform(0., vid.metadata[vid.current_stream]['duration'])
            else:
                start = random.uniform(0., 3600) # duration is an hour
            # seek only to keyframes
            vid.seek(start, stream="video", any_frame=False)
            frame, current_pts, stream_t = vid.next("video")
            if self.transform:
                frame = self.transform(frame)
                
            output = {
                    'path': path,
                    'video': frame,
                    'target': target
            }
            yield output