import os
import random
from glob import glob

import pandas as pd
import torch
from torchvision.datasets.folder import make_dataset
from torchvision import transforms as t

from video import Video


class AVADataset(torch.utils.data.IterableDataset):
    def __init__(self, root, annotation_file, clip_len=16, frame_transform=None, video_transform=None, frame_position="beginning"):
        super(AVADataset).__init__()
        self._preprocess_dataset(annotation_file)  # get dataset file
        self.clip_len = clip_len
        self.root=root
        self.frame_transform=frame_transform
        self.video_transform = video_transform
        assert frame_position in ["beginning", "middle"], "Keyframe can be either in the middle or at the beginning of the clip"
        self.fp = frame_position
        
    def _preprocess_dataset(self, annotation_file):
        df = pd.read_csv(annotation_file, names=["video_id", "middle_frame_timestamp", "x1", "y1", "x2", "y2", "action_id", "person_id"])
        # aggregate the actions/bounding-boxes according to have a list
        # of each for each video
        
        self.df = (df.groupby(['video_id', 'middle_frame_timestamp'], as_index=False)
      .agg(lambda x: list(x))).reset_index(drop=True)
        
    def __iter__(self):
        for idx in range(len(self.df)):
            video_id = self.df.video_id.loc[idx]
            start = self.df.middle_frame_timestamp.loc[idx]
            path = glob(os.path.join(self.root, video_id+"*"))[0]
            
            vid = Video(path, debug=False)
            video_frames = [] # video frame buffer
            
            if self.fp != "beginning":
                # FIXME: this is approximate situation
                fps = vid.metadata[vid.current_stream]['fps']
                start = max(0, start - float((self.clip_len // 2) * fps))

            vid.seek(start, stream="video", any_frame=True)
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
                    # These are a pain bc the size is not constant,
                    # How to handle this?
#                     'x1': self.df.x1.loc[idx],
#                     'y1': self.df.y1.loc[idx],
#                     'x2': self.df.x2.loc[idx],
#                     'y2': self.df.y2.loc[idx],
#                     'action_id': self.df.action_id.loc[idx],
                    'person_id': self.df.person_id.loc[idx],
            }
            yield output
                