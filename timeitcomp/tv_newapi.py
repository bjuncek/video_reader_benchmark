import argparse
import timeit
import os
import pandas as pd

import torch
import torchvision

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('n', type=int,
                    help='Number of trials to run')
args = parser.parse_args()


setup_newAPI = """\
import torch
import torchvision
"""


def next_list(path):
    reader = torch.classes.torchvision.Video(path, "video", True)
    frames = []
    t, _ = reader.next()
    while t.numel() > 0:
        frames.append(t)
        t, _ = reader.next()

        
loaders = []
times = []
video = []
num_frames = []
lib_version = []


for i in range(args.n):
    print(i)
    for file in os.listdir("../videos"):
        if file in ["README", ".ipynb_checkpoints", "WUzgd7C1pWA.mp4", "SOX5yA1l24A.mp4", "R6llTwEh07w.mp4"]:
            print("Skipping README")
            continue
          
        path = os.path.join("../videos/", file)
        print(path)
        reader = torch.classes.torchvision.Video(path, "video", True)
        frames = []
        t, _ = reader.next()
        while t.numel() > 0:
            frames.append(t)
            t, _ = reader.next()
        nframes = len(frames)
        print(nframes)
        
        times.append( timeit.timeit(f"next_list(\"{path}\")", setup=setup_newAPI, globals=globals(), number=args.n)/args.n)
        video.append(file)
        loaders.append("newAPI")
        num_frames.append(nframes)
        lib_version.append(torchvision.__version__)


        
df = pd.DataFrame({"loader": loaders, "video": video, "time":times, "num_frames":num_frames, "lib_version": lib_version})
df.to_csv("out/tv_newAPI.csv") 
        
        

