import argparse
import timeit
import os
import math
import pandas as pd

import decord
from decord import VideoReader, cpu, gpu



parser = argparse.ArgumentParser(description="Process some integers.")
parser.add_argument("n", type=int, help="Number of trials to run")
args = parser.parse_args()


setup_decord = """\
from decord import VideoReader, cpu, gpu
"""
def get_decord_cpu(path):
    images_av = []
    vr = VideoReader(path, ctx=cpu(0))
    for i in range(len(vr)):
        # the video reader will handle seeking and skipping in the most efficient manner
        images_av.append(vr[i])


def get_decord_gpu(path):
    images_av = []
    vr = VideoReader(path, ctx=gpu(0))
    for i in range(len(vr)):
        # the video reader will handle seeking and skipping in the most efficient manner
        images_av.append(vr[i])

loaders = []
times = []
times_random_seek = []
video = []
num_frames = []
lib_version = []


for i in range(args.n):
    for file in os.listdir("../videos"):
        if file in ["README", ".ipynb_checkpoints", "avadl.py"]:
            print(f"Skipping {file}")
            continue

        path = os.path.join("../videos/", file)
        images_av = []
        vr = VideoReader(path, ctx=gpu(0))
        for i in range(len(vr)):
            # the video reader will handle seeking and skipping in the most efficient manner
            images_av.append(vr[i])
        
        nframes = len(images_av)
        
        print(path, nframes)

        video.append(file)
        loaders.append("decord_gpu")
        num_frames.append(nframes)
        lib_version.append(decord.__version__)

        times.append(
            timeit.timeit(
                f'get_decord_gpu("{path}")',
                setup=setup_decord,
                globals=globals(),
                number=args.n,
            )
            / args.n
        ) 

        video.append(file)
        loaders.append("decord_cpu")
        num_frames.append(nframes)
        lib_version.append(decord.__version__)

        times.append(
            timeit.timeit(
                f'get_decord_cpu("{path}")',
                setup=setup_decord,
                globals=globals(),
                number=args.n,
            )
            / args.n
        )

df = pd.DataFrame(
    {
        "decoder": loaders,
        "video": video,
        "time": times,
        "num_frames": num_frames,
        "lib_version": lib_version,
    }
)
df.to_csv("out/decord_video.csv")
