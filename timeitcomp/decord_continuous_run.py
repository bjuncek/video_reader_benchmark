import os
from decord import VideoReader, VideoLoader
from decord import cpu, gpu


def get_decord_loader(path):
    vr = VideoLoader(path, shape=(20, 256, 340, 3), ctx=cpu(0), interval=0, skip=0, shuffle=1)
    for i in range(len(vr)):
        frames, _ = vr.next()
        
videos = [os.path.join("../videos/", x) for x in os.listdir("../videos") if x not in ["README", ".ipynb_checkpoints"]]


while True:
    get_decord_loader(videos)