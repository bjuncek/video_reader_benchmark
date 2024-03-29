import torch
import os

import torch
import torchvision
torchvision.set_video_backend("video_reader")

print("Imports over: video reader backend", torchvision.__version__)
videos = [os.path.join("../videos/", x) for x in os.listdir("../videos") if x not in ["README", ".ipynb_checkpoints"]]

print("Now starting video decoding for profile")
for i in range(100):
    for path in videos:
        frames = []
        reader = torchvision.io.VideoReader(path, "video")
        for data in reader:
            frames.append(data["data"])