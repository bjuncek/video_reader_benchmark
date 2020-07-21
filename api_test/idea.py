""" APIs needed
1. some form of interface with the video that provides frame-by-frame reading
2. Wrapper around it for clip-based datasets (Kinetics)
3. Wrapper around it for long videos (either frame or clip based)
3.1 Parallel decoding support
"""


"""
The overall idea iiuc is to have a simple wrapper around 
pyav that would support functionality we need. 

"""

class Video(object):
    def __init__(self, stream, width=-1, height=-1, num_decoding_threads=0):

        self.container # this needs
        self.keyframes = self._read_keyframes()  # list of keyframes in sec what is the better way
        self.len = self.__len__()
        self.duration = # would be usefull to have? 
        self.avg_fps
# this is expensive to get  
#     def __len__(self):
#         return get_num_frames(self.container)
# decord shows this is not a good idea
#     def __getitem__(self, start_time):
#         self.seek(start_time)
#         return self.next()

    def get_clip(self, start_time, end_time=-1):
        "maybe optimized version of the function above?"
        # push for later
    
    def _get_current_timestamp(self):
        self.container.ts

    def seek(self, ts):
        """Here the question is how we're going to handle this - do we want
        to seek into the closest keyframe, or?
        """
        self.container.seek(ts)
       
    def fast_seek(self, ts):
        """
        seek to the neighrest keyframe
        """
        
    def next(self):
        return self.container.next_frame()

# this is needed for "fast approximate seek"
    def _read_keyframes(self):
        """list all keyframes (without decoding full video,
        this would be needed to support fast reading for what tullie wanted
        """


        

