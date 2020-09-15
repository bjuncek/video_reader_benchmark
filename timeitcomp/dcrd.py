import argparse
import timeit
import os
import pandas as pd

import decord
from decord import VideoReader
from decord import cpu, gpu

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('n', type=int,
                    help='Number of trials to run')
args = parser.parse_args()

setup_decord = """\
from decord import VideoReader
from decord import cpu
"""

def get_decord(path):
    images_d = []
    vr = VideoReader(path, ctx=cpu(0))
    for i in range(len(vr)):
        # the video reader will handle seeking and skipping in the most efficient manner
        images_d.append(vr[i])



loaders = []
times = []
video = []
num_frames = []
lib_version = []



for i in range(args.n):
    for file in os.listdir("../videos"):
        if file in ["README", ".ipynb_checkpoints"]:
            print("Skipping README")
            continue

        
        path = os.path.join("../videos/", file)
        print(path)

        # get number of files using the decord
        images_d = []
        vr = VideoReader(path, ctx=cpu(0))
        for i in range(len(vr)):
            # the video reader will handle seeking and skipping in the most efficient manner
            images_d.append(vr[i])
        
        num_frames.append(len(images_d))
        times.append( timeit.timeit(f"get_decord(\"{path}\")", setup=setup_decord, globals=globals(), number=args.n)/args.n)
        video.append(file)
        loaders.append("decord")
        lib_version.append(1)

df = pd.DataFrame({"loader": loaders, "video": video, "time":times, "num_frames":num_frames, "lib_version": lib_version})
df.to_csv("out/decord.csv")