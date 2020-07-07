import torch
import os
import av
import cv2
import timeit
import numpy as np
import pandas as pd
import torchvision

NUBMER_TRIALS=10

setup_pyav = """\
import torch
import av
import numpy as np
"""

def get_pyav(path): 
    images_av = []
    container = av.open(path)
    # container.streams.video[0].thread_count = 1  # force single thread
    for frame in container.decode(video=0):
        images_av.append(frame.to_rgb().to_ndarray())


setup_cv2 = """\
import torch
import cv2
cv2.setNumThreads(1)
import numpy as np
"""

def get_cv2(path):
    images_cv2 = []
    cv2.setNumThreads(1)
    cap = cv2.VideoCapture(path)
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret is True:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            images_cv2.append(frame)
        else:
            break
    cap.release()

setup_tv = """\
import torch
import torchvision
torchvision.set_video_backend("pyav")
"""

def get_tv(path):
    vframes, _, _ = torchvision.io.read_video(path)

setup_tvvr = """\
import torch
import torchvision
torchvision.set_video_backend("video_reader")
"""

def get_tvvr(path):
    vframes, _, _ = torchvision.io.read_video(path)


loaders = []
times = []
video = []
for file in os.listdir("./videos"):
    if file in ["README", ".ipynb_checkpoints"]:
        print("Skipping README")
        continue
    path = os.path.join("./videos/", file)

    times.append( timeit.timeit(f"get_cv2(\"{path}\")", setup=setup_cv2, globals=globals(), number=NUBMER_TRIALS)/NUBMER_TRIALS)
    video.append(file)
    loaders.append("CV2")

    times.append( timeit.timeit(f"get_pyav(\"{path}\")", setup=setup_pyav, globals=globals(), number=NUBMER_TRIALS)/NUBMER_TRIALS)
    video.append(file)
    loaders.append("pyav")

    times.append( timeit.timeit(f"get_tv(\"{path}\")", setup=setup_tv, globals=globals(), number=NUBMER_TRIALS)/NUBMER_TRIALS)
    video.append(file)
    loaders.append("tv_pyav")

    times.append( timeit.timeit(f"get_tvvr(\"{path}\")", setup=setup_tvvr, globals=globals(), number=NUBMER_TRIALS)/NUBMER_TRIALS)
    video.append(file)
    loaders.append("tv_vr")

df = pd.DataFrame({"loader": loaders, "video": video, "time":times})
df.to_csv("pyav_basic_speeds.csv")