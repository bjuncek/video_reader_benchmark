import argparse
import timeit
import os
import pandas as pd

import itertools
import torchvision

parser = argparse.ArgumentParser(description="Process some integers.")
parser.add_argument("n", type=int, help="Number of trials to run")
args = parser.parse_args()



setup_tvvr = """\
import torch
import torchvision
torchvision.set_video_backend("video_reader")
"""

def measure_reading_video(path):
    vframes, _, _ = torchvision.io.read_video(path)


loaders = []
times_per_video = []
times_random_seek = []
video = []
num_frames = []
lib_version = []


for i in range(args.n):
    print(i)
    for file in os.listdir("../videos"):
        if file in ["README", ".ipynb_checkpoints", "avadl.py"]:
            print(f"Skipping {file}")
            continue

        path = os.path.join("../videos/", file)

        vframes, _, _ = torchvision.io.read_video(path)
        nframes = len(vframes)

        print(path, nframes)

        video.append(file)
        loaders.append("VideoReader")
        num_frames.append(nframes)
        lib_version.append(torchvision.__version__)

        times_per_video.append(
            timeit.timeit(
                f'measure_reading_video("{path}")',
                setup=setup_tvvr,
                globals=globals(),
                number=args.n,
            )
            / args.n
        )



df = pd.DataFrame(
    {
        "decoder": loaders,
        "video": video,
        "time": times_per_video,
        "num_frames": num_frames,
        "lib_version": lib_version,
    }
)
df.to_csv("out/tvOldApi_fullvideo.csv")

