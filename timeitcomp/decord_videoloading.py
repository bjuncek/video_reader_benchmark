import torch
import os

import timeit
import numpy as np
import pandas as pd
import torchvision
from decord import VideoReader
from decord import cpu

NUBMER_TRIALS=10


setup_decord = """\
from decord import VideoReader, VideoLoader
from decord import cpu, gpu
"""

def get_decord(path):
    vr = VideoReader(path, ctx=cpu(0))
    len_vr = len(vr)
    for i in range(len(vr)):
        # the video reader will handle seeking and skipping in the most efficient manner
        _ = vr[i]


def get_decord_batch(path):
    vr = VideoReader(path, ctx=cpu(0))
    len_vr = len(vr)
    batch = vr.get_batch(range(len_vr))

    
def get_decord_loader(path):
    vr = VideoLoader(path, shape=(20, 256, 340, 3), ctx=cpu(0), interval=0, skip=0, shuffle=1)
    for i in range(len(vr)):
        frames, _ = vr.next()



loaders = []
times = []

for i in range(10):
    videos = [os.path.join("../videos/", x) for x in os.listdir("../videos") if x not in ["README", ".ipynb_checkpoints"]]

    times.append(timeit.timeit(f"get_decord_loader({videos})", setup=setup_decord, globals=globals(), number=NUBMER_TRIALS)/NUBMER_TRIALS/len(videos))
    loaders.append("decord_VL")

    VR_FL_TIMES = []
    VR_B_TIMES = []
    
    for path in videos:
        VR_FL_TIMES.append( timeit.timeit(f"get_decord(\"{path}\")", setup=setup_decord, globals=globals(), number=NUBMER_TRIALS)/NUBMER_TRIALS)


        VR_B_TIMES.append( timeit.timeit(f"get_decord_batch(\"{path}\")", setup=setup_decord, globals=globals(), number=NUBMER_TRIALS)/NUBMER_TRIALS)
    
    times.append(np.mean(VR_FL_TIMES))
    loaders.append("decord_VR_forloop")
    
    times.append(np.mean(VR_B_TIMES))
    loaders.append("decord_VR_batch")
 
        


df = pd.DataFrame({"loader": loaders, "time":times})
df.to_csv("decord_dataloading_comp.csv")