import argparse
import timeit
import os
import pandas as pd

import cv2

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('n', type=int,
                    help='Number of trials to run')
args = parser.parse_args()


setup_cv2 = """\
import torch
import cv2
cv2.setNumThreads(1)
import numpy as np
"""

def get_cv2(path):
    images_cv2 = []
    cap = cv2.VideoCapture(path)
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret is True:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            images_cv2.append(frame)
        else:
            break
    cap.release()



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
        
        images_cv2 = []
        cap = cv2.VideoCapture(path)
        while(cap.isOpened()):
            ret, frame = cap.read()
            if ret is True:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                images_cv2.append(frame)
            else:
                break
        cap.release()
        nframes = len(images_cv2)

        times.append(timeit.timeit(f"get_cv2(\"{path}\")", setup=setup_cv2, globals=globals(), number=args.n)/args.n)
        video.append(file)
        loaders.append("CV2")
        num_frames.append(nframes)
        
        lib_version.append(cv2.__version__)

df = pd.DataFrame({"loader": loaders, "video": video, "time":times, "num_frames":num_frames, "lib_version": lib_version})
df.to_csv("out/cv2.csv")