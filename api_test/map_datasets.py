"""
Some key questions: 
1. how feasible would it be to keep all the video objects in memory? 
2. 

Notes:
    1. this approach in IMHO questionable. Can we somehow use IterableDataset
    to split the object accross replicas (so that it takes less memory), and then
"""
    

class Kinetics400(VisionDataset):

    def __init__(self, root, frames_per_clip, step_between_clips=1, frame_rate=None,
                    extensions=('avi',), transform=None, _precomputed_metadata=None,
                    num_workers=1):
            super(Kinetics400, self).__init__(root)

        classes = list(sorted(list_dir(root)))
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        self.samples = make_dataset(self.root, class_to_idx, extensions, is_valid_file=None)
        self.classes = classes
        video_list = [x[0] for x in self.samples]
        self.video_clips = VideoClips(
            video_list,
            frames_per_clip,
            step_between_clips,
            frame_rate,
            _precomputed_metadata,
            num_workers=num_workers,
        )
        self.transform = transform

    @property
    def metadata(self):
        return self.video_clips.metadata

    def __len__(self):
        return self.video_clips.num_clips()

    def __getitem__(self, idx):
        video, audio, info, video_idx = self.video_clips.get_clip(idx)
        label = self.samples[video_idx][1]

        if self.transform is not None:
            video = self.transform(video)

        return video, audio, label


class ActivityNetClips(VisionDataset):
    """ 
    Activity net dataset loaded as clips (often used)
    """
    def __init__(self, csv, frames_per_clip, step_between_clips=1, frame_rate=None,
                     transform=None, _precomputed_metadata=None,
                    num_workers=1):
            super(ActivityNetClips, self).__init__(root)

        self.df = pd.load_csv(csv)
        video_list = self.df.video_path.to_list()
        time_range = list(zip(self.df.start.to_list(), self.df.end.to_list()))
        self.video_clips = VideoClips(
            video_list,
            frames_per_clip,
            step_between_clips,
            frame_rate,
            _precomputed_metadata,
            num_workers=num_workers,
        )
        self.video_clips.subset_timerange(time_range)
        self.transform = transform

    @property
    def metadata(self):
        return self.video_clips.metadata

    def __len__(self):
        return self.video_clips.num_clips()

    def __getitem__(self, idx):
        video, audio, info, video_idx = self.video_clips.get_clip(idx)
        label = self.df.label.loc[video_idx]

        if self.transform is not None:
            video = self.transform(video)

        return video, audio, label


class ActivityNetVideos(object):
    """ 
    Activity net dataset loaded as videos (less ofted as can't be batched, there
    woudl be additional transform )
    """
    def __init__(self, csv,  transform=None,
                    num_workers=1):
        super(ActivityNetVideos, self).__init__()

        self.df = pd.load_csv(csv)
        self.videos = [Video(video) for video in self.df.video_path.to_list()]
        self.transform = transform
    
    def __len__(self):
        return len(self.df)

    def _get_video(self, idx):
        current_time = self.df.start.loc[idx]
        end = self.df.end.loc[idx]
        video = self.videos[idx]
        list_of_frames = []
        video.seek(current_time)
        while current_time <= end:
            data, current_time = video.next()
            list_of_frames.append(data)
        return torch.stack(list_of_frames)


    def __getitem__(self, idx):
        video, audio, whatever =  self._get_video(idx)

        return whatever we need
