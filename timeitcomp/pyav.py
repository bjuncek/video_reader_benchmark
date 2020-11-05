import argparse
import timeit
import os
import math
import pandas as pd

import av


parser = argparse.ArgumentParser(description="Process some integers.")
parser.add_argument("n", type=int, help="Number of trials to run")
args = parser.parse_args()


setup_pyav = """\
import torch
import av
import numpy as np
"""


def get_pyav_video(path):
    images_av = []
    container = av.open(path)
    container.streams.video[0].thread_count = 1  # force single thread
    for frame in container.decode(video=0):
        images_av.append(frame.to_rgb().to_ndarray())


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


loaders = []
times = []
times_random_seek = []
video = []
num_frames = []
lib_version = []

for i in range(args.n):
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
        loaders.append("pyav")
        num_frames.append(nframes)
        lib_version.append(av.__version__)

        times.append(
            timeit.timeit(
                f'get_pyav_video("{path}")',
                setup=setup_pyav,
                globals=globals(),
                number=args.n,
            )
            / args.n
        )

        times_random_seek.append(
            timeit.timeit(
                f"get_pyav_random_seek(container, time)",
                setup=setup_pyav,
                globals=globals(),
                number=args.n,
            )
            / args.n
        )


df = pd.DataFrame(
    {
        "decoder": loaders,
        "video": video,
        "time": times,
        "num_frames": num_frames,
        "lib_version": lib_version,
    }
)
df.to_csv("out/pyav_video.csv")


df = pd.DataFrame(
    {
        "decoder": loaders,
        "video": video,
        "time": times_random_seek,
        "num_frames": num_frames,
        "lib_version": lib_version,
    }
)
df.to_csv("out/pyav_random_seek.csv")