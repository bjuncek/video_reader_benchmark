import av
import math
import torchvision.transforms as t

class Video(object):
    def __init__(self, path_to_video, stream="video"):
        self.container = av.open(path_to_video, metadata_errors="ignore")
        
        # safety check
        avail_streams = [x.type for x in list(self.container.streams)]
        assert stream in avail_streams, "Stream not existent"
        # get correct stream
        stream_idx = avail_streams.index(stream)
        self.stream = list(self.container.streams)[stream_idx]

        # any additional preprocessing we want to do
        self._read_keyframes()
        
    
    def _read_keyframes(self):
        """list all keyframes (without decoding full video,
        this would be needed to support fast reading for what tullie wanted
        """
        self.keyframes = []
        #TODO: this kinda works
        for packet in self.container.demux():
            if packet.is_keyframe:
                self.keyframes.append(Video._stream_to_sec(packet.pts, self.stream))
        
    def seek(self, ts, any_frame=False):
        """ 
        any_Frame allows for a more precise seek
        """
        start_offset = Video._sec_to_stream(ts, self.stream)
        self.container.seek(start_offset, any_frame=any_frame, stream=self.stream)

    # def _next_from_stream(self,stream):
    #     """
    #     Helper to support multiple stream handling
    #     """
    #     return self.container.decode(stream).__next__()

    def next(self):
        """
        return a tuple of streams for as many streams as we want
        """
        frame = self.container.decode(self.stream).__next__()
        ts = Video._stream_to_sec(frame.pts, self.stream)
        image = frame.to_ndarray()
        return image, ts

    @staticmethod
    def _sec_to_stream(ts, stream):
        return int(math.floor(ts * (1 / stream.time_base)))
    
    @staticmethod
    def _stream_to_sec(pts, stream):
        return float(pts * stream.time_base)
        