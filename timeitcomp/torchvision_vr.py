import os
import argparse
import timeit
import pandas as pd

import torchvision


parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('n', type=int,
                    help='Number of trials to run')
args = parser.parse_args()



setup_tvvr = """\
import torch
import torchvision
torchvision.set_video_backend("video_reader")
"""

def get_tvvr(path):
    vframes, _, _ = torchvision.io.read_video(path, pts_unit="sec")


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

        vframes, _, _ = torchvision.io.read_video(path, pts_unit="sec")
        nframes = len(vframes)

        times.append( timeit.timeit(f"get_tvvr(\"{path}\")", setup=setup_tvvr, globals=globals(), number=args.n)/args.n)
        video.append(file)
        loaders.append("tv_videoreader")
        num_frames.append(nframes)
        lib_version.append(torchvision.__version__)
        
df = pd.DataFrame({"loader": loaders, "video": video, "time":times, "num_frames":num_frames, "lib_version": lib_version})
df.to_csv("out/tv_pyav.csv") 