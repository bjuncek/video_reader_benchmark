import av
import math
import warnings
import torchvision.transforms as t

class Video(object):
    def __init__(self, path_to_video, stream="video", debug=True):

        self.container = av.open(path_to_video, metadata_errors="ignore")
        self.debug = debug

        # any additional preprocessing we want to do
        self._init_stream_dict()
        self._get_metadata()
        self._set_current_stream(stream)

        if self.debug:
            print(path_to_video, "\n \t default stream: ", self.current_stream)
    
    def _get_metadata(self):
        # TODO: docs
        metadata = {}
        for key in self.available_streams.keys():
            for stream in self.available_streams[key]:
                metadata[stream] = {"fps": stream.rate}
                if stream.duration or self.container.duration:
                    duration = self.container.duration if not stream.duration else stream.duration
                    metadata[stream]["duration"] = duration * stream.time_base
                else:
                    metadata[stream]["duration"] = -1
        self.metadata = metadata
    
    def _init_stream_dict(self):
        # TODO: DOCS
        self.available_streams = {}
        for stream in list(self.container.streams):
            if stream.type not in self.available_streams:
                self.available_streams[stream.type] = [stream]
            else:
                self.available_streams[stream.type].append(stream)
        if self.debug:
            _ = self.list_streams()
        
    def list_streams(self):
        # TODO: docs
        if self.debug:
            # TODO: modify repr
            print("List of available streams:")
            print(self.available_streams)
        return self.available_streams
        
    def _set_current_stream(self, stream):
        #TODO: docs
        avail_stream_types = [x.type for x in list(self.container.streams)]
        st_type, st_idx = stream, 0
        if stream.rfind(":") > 0:
            i = stream.rfind(":")
            st_type, st_idx = stream[:i], int(stream[i+1:])
        if st_idx < 0:
            # TODO: verify if thisi is a correct assumption
            warnings.warn("Negative stream index defaults to 0")
            st_idx = 0
        
        assert st_type in avail_stream_types, f"Stream type not exists {st_type} / {avail_stream_types}"
        assert st_idx < len(self.available_streams[st_type]), f"Stream idx out of bounds {st_idx} / {len(self.available_streams[st_type])}"
        # sett the correct stream from the dict
        self.current_stream = self.available_streams[st_type][st_idx]
    
    def peak(self, ts, stream=None, backward=True, any_frame=False):
        if any_frame:
            print("This could be slow")
        if stream is not None:
            self._set_current_stream(stream)
        
        self.seek(ts, stream, backward, any_frame)
        packet = next(self.container.demux(self.current_stream))
        current_pts = packet.pts
        is_kf = packet.is_keyframe
        self.seek(ts, stream, backward, any_frame)
        return (current_pts, Video._stream_to_sec(current_pts), is_kf)
        
    def seek(self, ts, stream=None, backward=True, any_frame=False):
        # TODO: docs
        if stream is not None:
            self._set_current_stream(stream)
        if self.metadata[self.current_stream]['duration'] > 0 and ts > self.metadata[self.current_stream]['duration']:
            warnings.warn(f"Seeking to {ts}, video duration is {self.metadata[self.current_stream]['duration']} - will seek to the last available keyframe")
            self.container.seek(start_offset, backward=backward, any_frame=False, stream=self.current_stream)
            return
        
        start_offset = Video._sec_to_stream(ts, self.current_stream)
        self.container.seek(start_offset, backward=backward, any_frame=False, stream=self.current_stream)
        
        if any_frame:
            # we get the timestamp of the current frame which is
            # the first keyframe before the index we asked for
            _, curr_ts, _ = self.next()
            # NOTE: this is an estimate - we're assuming that the next frame will be located
            #       at 1/fps s away from the current frame. We then decode packets until the
            #       just before the place we want to seek. Ideally, we'd want the following
            #       `next()` to return the closest frame to the one we've been looking for
            per_frame_time = float(1/self.metadata[self.current_stream]['fps'])
            next_hat = curr_ts + per_frame_time
            while next_hat < ts:
                prev_ts = curr_ts
                _, curr_ts, _ = self.next()
                next_hat = curr_ts + per_frame_time
                if self.debug:
                    print("Empirical_next_delta:", curr_ts - prev_ts, 
                          "\n Estimated_next_delta:", next_hat-curr_ts,
                          "\n Estimated_next", next_hat, "\n Current_ts", curr_ts,
                          "\n Requested_ts", ts, "\n")

    def next(self, stream=None):
        #TODO: docs
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
        
