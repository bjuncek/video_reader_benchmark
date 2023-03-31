import av
import os
import pandas as pd
import timeit


parser = argparse.ArgumentParser(description="Process some integers.")
parser.add_argument("n", type=int, help="Number of trials to run")
args = parser.parse_args()

setup_pyav = """\
import torch
import av
import numpy as np
"""

def get_pyav_default(path): 
    images_av = []
    container = av.open(path)
    # container.streams.video[0].thread_type =  # force single thread
    for frame in container.decode(video=0):
        images_av.append(frame.to_rgb().to_ndarray())

def get_pyav(path, ttype): 
    images_av = []
    container = av.open(path)
    container.streams.video[0].thread_type = ttype # force single thread
    for frame in container.decode(video=0):
        images_av.append(frame.to_rgb().to_ndarray())


videos = [os.path.join("../videos/", x) for x in os.listdir("./videos") if x not in ["README", ".ipynb_checkpoints"]]

thread_type = []
times = []
video = []
for i in range(10):
    for path in videos:
        thread_type.append("default")
        times.append(timeit.timeit(f"get_pyav_default(\"{path}\")", setup=setup_pyav, globals=globals(), number=args.n)/args.n)
        video.append(path)
        for t in ["AUTO", "SLICE", "FRAME"]:
            thread_type.append(t)
            video.append(path)
            times.append(timeit.timeit(f"get_pyav(\"{path}\", \"{t}\")", setup=setup_pyav, globals=globals(), number=args.n)/args.n)

df = pd.DataFrame({"threading": thread_type, "video": video, "time":times})
df.to_csv("pyav_threading_speeds.csv")