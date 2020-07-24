import av
import math
import torchvision.transforms as t

class Video(object):
    def __init__(self, path_to_video, video=True, audio=False, width=-1, height=-1, num_decoding_threads=0):
        self.container = av.open(path_to_video, metadata_errors="ignore")
        self._read_keyframes()
        self.video_flag = video
        self.audio_flag = audio
        transforms = [t.ToTensor()]
        if width > 0 and height>0:
            transforms.insert(0, t.Resize((height, width), interpolation=2))
        self.transform = t.Compose(transforms)
        
    
    def _read_keyframes(self):
        """list all keyframes (without decoding full video,
        this would be needed to support fast reading for what tullie wanted
        """
        self.keyframes = []
        #TODO: this kinda works
        for packet in self.container.demux():
            if packet.is_keyframe:
                self.keyframes.append(Video._stream_to_sec(packet.pts, self.container.streams.video[0]))
        
    def seek(self, ts, any_frame=False, stream=None):
        """ 
        any_Frame allows for a more precise seek
        """
        # TODO: check if this is correct assumption
        if not stream:
            stream = self.container.streams.video[0]
        start_offset = Video._sec_to_stream(ts, stream)
        self.container.seek(start_offset, any_frame=any_frame, stream=stream)

    def _next_from_stream(self,stream):
        """
        Helper to support multiple stream handling
        """
        return self.container.decode(stream).__next__()

    def next(self):
        """
        return a tuple of streams for as many streams as we want
        """
        frame = self._next_from_stream(self.container.streams.video[0])
        ts = Video._stream_to_sec(frame.pts, self.container.streams.video[0])
        image = self.transform(frame.to_image())

        return image, ts

    @staticmethod
    def _sec_to_stream(ts, stream):
        return int(math.floor(ts * (1 / stream.time_base)))
    
    @staticmethod
    def _stream_to_sec(pts, stream):
        return float(pts * stream.time_base)
        