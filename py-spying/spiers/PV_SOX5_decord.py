import os
from decord import VideoReader
from decord import cpu, gpu


path = "../videos/SOX5yA1l24A.mp4"

for i in range(1000):
    images_d = []
    vr = VideoReader(path, ctx=cpu(0))
    for i in range(len(vr)):
        # the video reader will handle seeking and skipping in the most efficient manner
        images_d.append(vr[i])

print(len(images_d))