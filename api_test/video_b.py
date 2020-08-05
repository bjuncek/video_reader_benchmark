import av
import math
import torchvision.transforms as t

class Video(object):
    def __init__(self, path_to_video):
        self.container = av.open(path_to_video, metadata_errors="ignore")
        
        # safety check
        avail_streams = [x.type for x in list(self.container.streams)]
        assert "video" in avail_streams, "Video stream not existent"
        # get video stream idx
        self.video_idx = avail_streams.index("video")
        # self.stream = list(self.container.streams)[stream_idx]

        # any additional preprocessing we want to do
        self._read_keyframes()
        
    
    def _read_keyframes(self):
        """list all keyframes (without decoding full video,
        this would be needed to support fast reading for what tullie wanted
        """
        self.keyframes = []
        #TODO: this kinda works
        stream = list(self.container.streams)[self.video_idx]
        for packet in self.container.demux():
            if packet.is_keyframe:
                self.keyframes.append(Video._stream_to_sec(packet.pts, stream))
        
    def seek(self, ts, any_frame=False):
        """ 
        any_Frame allows for a more precise seek
        """
        ## NOTE: we,re always seeking into the video stream
        stream = list(self.container.streams)[self.video_idx]
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
        output = []
        for stream in list(self.container.streams):
            
            frame = self.container.decode(stream).__next__()
            ts = Video._stream_to_sec(frame.pts, stream)
            image = frame.to_ndarray()
            print(stream, ts)
            output.append((image, ts, stream.type))
        return output

    @staticmethod
    def _sec_to_stream(ts, stream):
        return int(math.floor(ts * (1 / stream.time_base)))
    
    @staticmethod
    def _stream_to_sec(pts, stream):
        return float(pts * stream.time_base)
        