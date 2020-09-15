import argparse
import timeit
import os
import pandas as pd


import av


parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('n', type=int,
                    help='Number of trials to run')
args = parser.parse_args()


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
        
        images_av = []
        container = av.open(path)
        container.streams.video[0].thread_count = 1  # force single thread
        for frame in container.decode(video=0):
            images_av.append(frame.to_rgb().to_ndarray())
        nframes =  len(images_av)
        
        
        times.append( timeit.timeit(f"get_pyav(\"{path}\")", setup=setup_pyav, globals=globals(), number=args.n)/args.n)
        video.append(file)
        loaders.append("pyav")
        num_frames.append(nframes)
        lib_version.append(av.__version__)
        
        

df = pd.DataFrame({"loader": loaders, "video": video, "time":times, "num_frames":num_frames, "lib_version": lib_version})
df.to_csv("out/pyav.csv")