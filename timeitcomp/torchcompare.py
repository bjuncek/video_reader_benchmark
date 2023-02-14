import argparse
import os
import pandas as pd

import torchvision
import torch.utils.benchmark as benchmark

import cv2
import av
from decord import VideoReader, cpu, gpu



parser = argparse.ArgumentParser(description="Process some integers.")
parser.add_argument("--n", default=4, type=int, help="Number of trials to run")
parser.add_argument("--output", action="store_true", help="Output the results to a csv file", default=False)
args = parser.parse_args()


setup_decord = """\
from decord import VideoReader, cpu  #, gpu
"""

setup_newAPI = """\
import torch
import torchvision
"""

setup_tvvr = """\
import torch
import torchvision
torchvision.set_video_backend("video_reader")
"""

setup_tvpyav = """\
import torch
import torchvision
torchvision.set_video_backend("pyav")
"""

setup_cv2 = """\
import torch
import cv2
cv2.setNumThreads(1)
import numpy as np
"""

setup_pyav = """\
import torch
import av
import numpy as np
"""


def measure_PYAV(path):
    images_av = []
    container = av.open(path)
    container.streams.video[0].thread_count = 1  # force single thread
    for frame in container.decode(video=0):
        images_av.append(frame.to_rgb().to_ndarray())

def get_cv2(path):
    images_cv2 = []
    cap = cv2.VideoCapture(path)
    while cap.isOpened():
        ret, frame = cap.read()
        if ret is True:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            images_cv2.append(frame)
        else:
            break
    cap.release()

def measure_TVVR(path):
    vframes, _, _ = torchvision.io.read_video(path)

def measure_TV(path, threads=0):
    """We pass a path and get the
    loaded the video
    """
    frames = []
    reader = torchvision.io.VideoReader(path, "video", num_threads=threads)
    for data in reader:
        frames.append(data["data"])

def measure_DECORD(path):
    images_av = []
    vr = VideoReader(path, ctx=cpu(0))
    for i in range(len(vr)):
        # the video reader will handle seeking and skipping in the most efficient manner
        images_av.append(vr[i])

# def get_decord_gpu(path):
#     images_av = []
#     vr = VideoReader(path, ctx=gpu(0))
#     for i in range(len(vr)):
#         # the video reader will handle seeking and skipping in the most efficient manner
#         images_av.append(vr[i])


loaders = []
times = []
mean_times = []
video = []
num_frames = []
threads = []

for num_threads in [1, 4, 8]:
    for file in os.listdir("../videos"):
        if file in ["README", ".ipynb_checkpoints", "avadl.py"]:
            print(f"Skipping {file}")
            continue

        path = os.path.join("../videos/", file)
        images_av = []
        vr = VideoReader(path, ctx=cpu(0))
        for i in range(len(vr)):
            # the video reader will handle seeking and skipping in the most efficient manner
            images_av.append(vr[i])
        
        nframes = len(images_av)
        
        print(path, nframes)
        video.append(file)
        loaders.append("decord_cpu")
        num_frames.append(nframes)
        times.append(benchmark.Timer(
                stmt=f'measure_DECORD("{path}")',
                setup=setup_decord,
                globals=globals(),
                label="Video Reading",
                sub_label=str(file),
                description="DECORD",
                num_threads=num_threads,
            ).timeit(args.n))
        mean_times.append(times[-1].mean)
        threads.append(num_threads)

        
        video.append(file)
        loaders.append("tv_newAPI")
        num_frames.append(nframes)
        times.append(
            benchmark.Timer(
                stmt=f'measure_TV("{path}", {num_threads})',
                setup=setup_newAPI,
                globals=globals(),
                label="Video Reading",
                sub_label=str(file),
                description="TV-newAPI",
                num_threads=num_threads,
            ).timeit(args.n))
        mean_times.append(times[-1].mean)
        threads.append(num_threads)

        video.append(file)
        loaders.append("pyav")
        num_frames.append(nframes)
        times.append(
            benchmark.Timer(
                stmt=f'measure_PYAV("{path}")',
                setup=setup_pyav,
                globals=globals(),
                label="Video Reading",
                sub_label=str(file),
                description="pyav",
                num_threads=num_threads,
            ).timeit(args.n))
        mean_times.append(times[-1].mean)
        threads.append(num_threads)

        video.append(file)
        loaders.append("tv_vr")
        num_frames.append(nframes)
        times.append(
            benchmark.Timer(
                stmt=f'measure_TVVR("{path}")',
                setup=setup_tvvr,
                globals=globals(),
                label="Video Reading",
                sub_label=str(file),
                description="TV-vr",
                num_threads=num_threads,
            ).timeit(args.n))
        mean_times.append(times[-1].mean)
        threads.append(num_threads)

        video.append(file)
        loaders.append("tv_pyav")
        num_frames.append(nframes)
        times.append(
            benchmark.Timer(
                stmt=f'measure_TVVR("{path}")',
                setup=setup_tvpyav,
                globals=globals(),
                label="Video Reading",
                sub_label=str(file),
                description="TV-pyav",
                num_threads=num_threads,
            ).timeit(args.n))
        mean_times.append(times[-1].mean)
        threads.append(num_threads)

        video.append(file)
        loaders.append("openCV")
        num_frames.append(nframes)
        times.append(
            benchmark.Timer(
                stmt=f'get_cv2("{path}")',
                setup=setup_cv2,
                globals=globals(),
                label="Video Reading",
                sub_label=str(file),
                description="openCV",
                num_threads=num_threads,
            ).timeit(args.n))
        mean_times.append(times[-1].mean)
        threads.append(num_threads)

compare = benchmark.Compare(times)
compare.print()
if args.output:
    df = pd.DataFrame(
        {
            "decoder": loaders,
            "video": video,
            "time": mean_times,
            "num_frames": num_frames,
            "num_threads": threads,
        }
    )
    df.to_csv("out/MASTER.csv")
