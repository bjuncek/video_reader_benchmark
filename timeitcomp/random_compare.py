import argparse
import os
import pandas as pd

import torchvision
import torch.utils.benchmark as benchmark

import itertools
import av
import math
from decord import VideoReader, cpu



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


def get_pyav_random_seek(container, time, num_frames=10):
    stream = container.streams.video[0]
    stream_name = {"video": 0}

    start_offset = int(math.floor(time * (1 / stream.time_base)))
    container.seek(start_offset, any_frame=False, backward=True, stream=stream)

    frames = {}
    for _idx, frame in enumerate(container.decode(**stream_name)):
        if frame.pts > start_offset:
            frames[frame.pts] = frame

        if len(frames) >= num_frames:
            break

    result = [frames[i] for i in sorted(frames)]
    result = [frame.to_rgb().to_ndarray() for frame in result]


def measure_seek_and_decode(reader, seek_in_s, num_frames=10):
    frames = []
    for data in itertools.islice(reader.seek(seek_in_s), num_frames):
        frames.append(data["data"])

def measure_decord_one(reader, start_frame, num_frames=10):
    frames = reader.get_batch(list(range(start_frame,start_frame + num_frames))).asnumpy()
    
    


loaders = []
times = []
mean_times = []
video = []
num_frames = []
threads = []

for num_threads in [1,]:
    for file in os.listdir("../videos"):
        if file in ["README", ".ipynb_checkpoints", "avadl.py"]:
            print(f"Skipping {file}")
            continue

        path = os.path.join("../videos/", file)
        images_av = []
        container = av.open(path)
        container.streams.video[0].thread_count = 1  # force single thread
        for frame in container.decode(video=0):
            images_av.append(frame.to_rgb().to_ndarray())
        nframes = len(images_av)

        container = av.open(path)
        duration = float(
            container.streams.video[0].duration * container.streams.video[0].time_base
        )
        time = duration / 2
        print(path, nframes, duration)


        video.append(file)
        loaders.append("pyav_RS")
        num_frames.append(nframes)
        times.append(benchmark.Timer(
                stmt=f"get_pyav_random_seek(container, time)",
                setup=setup_pyav,
                globals=globals(),
                label="Video Reading",
                sub_label=str(file),
                description="PYAV (precise)",
                num_threads=num_threads,
            ).timeit(args.n))
        mean_times.append(times[-1].mean)
        threads.append(num_threads)

        reader = torchvision.io.VideoReader(path, "video")
        video.append(file)
        loaders.append("tv_NEWAPI_RS")
        num_frames.append(nframes)
        times.append(benchmark.Timer(
                stmt=f"measure_seek_and_decode(reader, time)",
                setup=setup_tvvr,
                globals=globals(),
                label="Video Reading",
                sub_label=str(file),
                description="TV_NEWAPI",
                num_threads=num_threads,
            ).timeit(args.n))
        mean_times.append(times[-1].mean)
        threads.append(num_threads)

        vr = VideoReader(path, ctx=cpu(0))
        start_frm = len(vr) // 2
        video.append(file)
        loaders.append("decord_one_RS")
        num_frames.append(nframes)
        times.append(benchmark.Timer(
                stmt=f"measure_decord_one(vr, start_frm)",
                setup=setup_decord,
                globals=globals(),
                label="Video Reading",
                sub_label=str(file),
                description="DECORD_A",
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
    df.to_csv("out/MASTER_RANDOM.csv")
