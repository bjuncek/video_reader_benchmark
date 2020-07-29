import av
import math
import warnings
import torchvision.transforms as t

class Video(object):
    def __init__(self, path_to_video, stream="video", debug=True):

        self.container = av.open(path_to_video, metadata_errors="ignore")
        self.fps = float(self.container.streams.video[0].framerate)
        self.debug = debug

        # any additional preprocessing we want to do
        self._init_stream_list()
        self._get_duration()
        self._set_current_stream(stream)
        self._read_keyframes()

        if self.debug:
            print(path_to_video, "\n \t default stream: ", self.current_stream)
    
    def _get_duration(self):
        duration = []
        for d in self.available_streams:
            duration.append(float(d['stream'].duration * d['stream'].time_base))
        self.duration = min(duration)
    
    def _init_stream_list(self):
        self.available_streams = [{"type": stream.type, "stream": stream} for stream in list(self.container.streams)]
        if self.debug:
            _ = self.list_streams()
    
    @property
    def length(self):
        return self.duration
            
    def list_streams(self):
        if self.debug:
            print("List of available streams: (id, stream_type, stream)")
            for i in range(len(self.available_streams)):
                print(f"\t {i}, {self.available_streams[i]['type']}, {self.available_streams[i]['stream']} ")
        return self.available_streams
        
    def _set_current_stream(self, stream):  
        
        if isinstance(stream, str):
            warnings.warn("Stream given as a descriptive string, will return the first stream of that type if it exists")
            avail_streams = [x.type for x in list(self.container.streams)]
            assert stream in avail_streams, "Stream not exists"
            stream_idx = avail_streams.index(stream)
            self.current_stream = list(self.container.streams)[stream_idx]
        elif isinstance(stream, int):
            assert stream >= 0 and stream < len(list(self.container.streams)), "Stream index out of bounds"
            self.current_stream = list(self.container.streams)[stream]
        elif isinstance(stream, av.stream.Stream):
            warnings.warn("Stream passed directly - if it doesn't exist it will fail ungracefully")
            self.current_stream = stream
        else:
            raise Exception("No streams defined")
    
    def _read_keyframes(self):
        """list all keyframes (without decoding full video,
        this would be needed to support fast reading for what tullie wanted
        """
        self.keyframes = []
        #TODO: this kinda works
        for packet in self.container.demux():
            if packet.is_keyframe:
                self.keyframes.append(Video._stream_to_sec(packet.pts, self.current_stream))
        
    def seek(self, ts, stream=None, backward=True, any_frame=False):
        """ 
        any_Frame allows for a more precise seek
        """
        if stream is not None:
            self._set_current_stream(stream)
        if ts > self.duration:
            warnings.warn(f"Seeking to {ts}, video duration is {self.duration} - failing silently and seeking to 0")
            ts = 0
        start_offset = Video._sec_to_stream(ts, self.current_stream)
        self.container.seek(start_offset, backward=backward, any_frame=any_frame, stream=self.current_stream)

    def next(self, stream=None):
        """
        return a tuple of streams for as many streams as we want
        """
        if stream is not None:
            self._set_current_stream(stream)
        
        ts = float("inf")
        image = []
        
        try:
            frame = next(self.container.decode(self.current_stream))
            ts = Video._stream_to_sec(frame.pts, self.current_stream)
            if self.current_stream.type == "video":
                frame = frame.to_rgb()
            image = frame.to_ndarray()
        except StopIteration:
            warnings.warn("Stopping")
            pass
        except av.AVError:
            warnings.warn("Couldn't read stuff")
            pass
        
        
        return image, ts, self.current_stream.type

    @staticmethod
    def _sec_to_stream(ts, stream):
        return int(math.floor(ts * (1 / stream.time_base)))
    
    @staticmethod
    def _stream_to_sec(pts, stream):
        return float(pts * stream.time_base)
        
