import torch
import os

import timeit
import numpy as np
import pandas as pd
import torchvision
from decord import VideoReader
from decord import cpu, gpu

NUBMER_TRIALS=10


setup_decord = """\
from decord import VideoReader
from decord import cpu, gpu
"""

def get_decord(path):
    images_d = []
    vr = VideoReader(path, ctx=cpu(0))
    len_vr = len(vr)
    for i in range(len(vr)):
        # the video reader will handle seeking and skipping in the most efficient manner
        images_d.append(vr[i])
    print("decord", len(images_d))

def get_decord_batch(path):
    vr = VideoReader(path, ctx=cpu(0))
    len_vr = len(vr)
    batch = vr.get_batch(range(len_vr))
    print("decord", batch.shape)   



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


        times.append( timeit.timeit(f"get_decord(\"{path}\")", setup=setup_decord, globals=globals(), number=NUBMER_TRIALS)/NUBMER_TRIALS)
        video.append(file)
        loaders.append("decord_forlooop")
        num_frames.append(nframes)

        times.append( timeit.timeit(f"get_decord_batch(\"{path}\")", setup=setup_decord, globals=globals(), number=NUBMER_TRIALS)/NUBMER_TRIALS)
        video.append(file)
        loaders.append("decord_batch")
        num_frames.append(nframes)

df = pd.DataFrame({"loader": loaders, "video": video, "time":times, "num_frames":num_frames})
df.to_csv("decord_speed_comp.csv")