import torch
import os
import av
import cv2
import timeit
import numpy as np
import pandas as pd
import torchvision
from decord import VideoReader
from decord import cpu, gpu

NUBMER_TRIALS=10

setup_pyav = """\
import torch
import av
import numpy as np
"""

def get_pyav(path): 
    images_av = []
    container = av.open(path)
    container.streams.video[0].thread_count = 1  # force single thread
    for frame in container.decode(video=0):
        images_av.append(frame.to_rgb().to_ndarray())
    print("PYAV", len(images_av))

def get_pyavnononsense(path): 
    images_av = []
    container = av.open(path)
    container.streams.video[0].thread_count = 1  # force single thread
    for frame in container.decode(video=0):
        images_av.append(frame)
    print("PYAV", len(images_av))



loaders = []
times = []
video = []
num_frames = []
tv_version = []

for i in range(10):
    for file in os.listdir("../videos"):
        if file in ["README", ".ipynb_checkpoints"]:
            print("Skipping README")
            continue

        
        path = os.path.join("../videos/", file)
        print(path)
        vframes, _, _ = torchvision.io.read_video(path)
        nframes = len(vframes)
        # tv_version.append(torchvision.__version__)




        times.append( timeit.timeit(f"get_pyavnononsense(\"{path}\")", setup=setup_pyav, globals=globals(), number=NUBMER_TRIALS)/NUBMER_TRIALS)
        video.append(file)
        loaders.append("pyav_notorgb")
        num_frames.append(nframes)
        tv_version.append(torchvision.__version__)

        times.append( timeit.timeit(f"get_tv(\"{path}\")", setup=setup_tv, globals=globals(), number=NUBMER_TRIALS)/NUBMER_TRIALS)
        video.append(file)
        loaders.append("tv_pyav")
        num_frames.append(nframes)
        tv_version.append(torchvision.__version__)



df = pd.DataFrame({"loader": loaders, "video": video, "time":times, "num_frames":num_frames, "tv": tv_version})
df.to_csv("basic_reading_speeds.csv")