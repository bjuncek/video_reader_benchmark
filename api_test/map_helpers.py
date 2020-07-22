"""
Some key questions: 
1. how feasible would it be to keep all the video objects in memory? 
2. 

Notes:
    1. this approach in IMHO questionable. Can we somehow use IterableDataset
    to split the object accross replicas (so that it takes less memory), and then
"""
class VideoClips(object):
    def __init__(self,
                 video_paths,  # a list of video objects
                 clip_length_in_frames=16,
                 frames_between_clips=1,
                 fps=None,
                 keyframe_only=False):
        
        self.num_frames = clip_length_in_frames
        # is it feasible to compute this in advance?
        # think about doind it in  parallel (inside dataloader)
        self.videos = [Video(path, width, height) for path in all_videos]
        if keyframe_only:
            self.pts = [video.keyframes for video in videos]
        else:
            self.pts = [video.get_all_pts() for video in videos]
            
        self.compute_clips(clip_length_in_frames, frames_between_clips, fps)
    
    @staticmethod
    def compute_clips_for_video(video_pts, num_frames, step, fps, frame_rate):
        if fps is None:
            # if for some reason the video doesn't have fps (because doesn't have a video stream)
            # set the fps to 1. The value doesn't matter, because video_pts is empty anyway
            fps = 1
        if frame_rate is None:
            frame_rate = fps
        total_frames = len(video_pts) * (float(frame_rate) / fps)
        idxs = VideoClips._resample_video_idx(
            int(math.floor(total_frames)), fps, frame_rate
        )
        video_pts = video_pts[idxs]
        clips = unfold(video_pts, num_frames, step)
        if isinstance(idxs, slice):
            idxs = [idxs] * len(clips)
        else:
            idxs = unfold(idxs, num_frames, step)
        return clips, idxs
    
    def subset_timerange(self, timerange):
        # for the sake of argument let's say that timerange is 
        # [(start, end)]
        assert len(self.pts) == len(timestep) "Each video should have a timestep"
        # there has to be a better way to do this
        for i in range(len(timestep)):
            self.pts[i] = [x for x in self.pts[i] if x > timerange[i][0] and x < timerange[i][1]]

        
    def compute_clips(self, num_frames, step, frame_rate=None):
        """
        Compute all consecutive sequences of clips from video_pts.
        Always returns clips of size `num_frames`, meaning that the
        last few frames in a video can potentially be dropped.
        Arguments:
            num_frames (int): number of frames for the clip
            step (int): distance between two clips
        """
        self.num_frames = num_frames
        self.step = step
        self.frame_rate = frame_rate
        self.clips = []
        self.resampling_idxs = []
        for video_pts, fps in zip(self.video_pts, self.video_fps):
            clips, idxs = self.compute_clips_for_video(
                video_pts, num_frames, step, fps, frame_rate
            )
            self.clips.append(clips)
            self.resampling_idxs.append(idxs)
        clip_lengths = torch.as_tensor([len(v) for v in self.clips])
        self.cumulative_sizes = clip_lengths.cumsum(0).tolist()

    def get_clip_location(self, idx):
        """
        Converts a flattened representation of the indices into a video_idx, clip_idx
        representation.
        """
        video_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if video_idx == 0:
            clip_idx = idx
        else:
            clip_idx = idx - self.cumulative_sizes[video_idx - 1]
        return video_idx, clip_idx

    @staticmethod
    def _resample_video_idx(num_frames, original_fps, new_fps):
        step = float(original_fps) / new_fps
        if step.is_integer():
            # optimization: if step is integer, don't need to perform
            # advanced indexing
            step = int(step)
            return slice(None, None, step)
        idxs = torch.arange(num_frames, dtype=torch.float32) * step
        idxs = idxs.floor().to(torch.int64)
        return idxs
    
        def __len__(self):
        return self.num_clips()

    def num_videos(self):
        return len(self.video_paths)

    def num_clips(self):
        """
        Number of subclips that are available in the video list.
        """
        return self.cumulative_sizes[-1]

    def get_clip(self, idx):
        video_idx, clip_idx = self.get_clip_location(idx)
        video = self.videos[video_idx]
        clip_pts = self.clips[video_idx][clip_idx]

        data = allocate_tensor(self.num_frames, len(video.streams))
        
        try:
            video.seek(clip_pts[0])
            for i in range(len(clip_pts)):
                data[i, ...], _  = video.next()
        except av.AVError:
            # TODO raise a warning?
            pass
        
        return data, video.metadata
        